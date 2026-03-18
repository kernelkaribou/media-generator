#!/usr/bin/env python
"""
Batch Poster Generator

Generates movie poster images for movies that don't have them.
Queries the media-generator API for movies missing posters,
generates image prompts using the existing AI text model,
sends them to InvokeAI for image generation, and uploads results.

Usage:
    python batch_poster_generate.py
    python batch_poster_generate.py --limit 10 --verbose
    python batch_poster_generate.py --media-api http://localhost:8000 --invokeai http://localhost:9090
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image

# Add project root to path for lib imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.aoai_model import aoaiText
from lib.ollama_model import ollamaText
from lib.local_openai_model import localOpenAIText
from lib.process_helper import processHelper

# Defaults
DEFAULT_MEDIA_API_URL = "http://localhost:8000"
DEFAULT_INVOKEAI_URL = "http://localhost:9090"
DEFAULT_PROMPTS_FILE = "/data/batch-prompts.json"
POLL_INTERVAL = 5
GENERATION_TIMEOUT = 300

# Model names to look up from InvokeAI
MAIN_MODEL_NAME = "FLUX.2 Klein 4B (GGUF Q4)"
VAE_MODEL_NAME = "FLUX.2 VAE"
ENCODER_MODEL_NAME = "FLUX.2 Klein Qwen3 4B Encoder"

# Image dimensions for poster generation (portrait aspect ratio)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1792

# Thumbnail dimensions and quality
THUMB_WIDTH = 512
THUMB_HEIGHT = 896
THUMB_QUALITY = 80


def _random_id(length=10):
    """Generate a random alphanumeric ID for graph node suffixes."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def lookup_invokeai_models(invokeai_url):
    """
    Query InvokeAI for installed models and return the three required model records.

    Returns a dict with keys: 'main', 'vae', 'qwen3_encoder'.
    """
    resp = requests.get(f"{invokeai_url}/api/v2/models/")
    resp.raise_for_status()
    data = resp.json()
    models = data.get("models", data) if isinstance(data, dict) else data

    found = {}
    for model in models:
        name = model.get("name", "")
        if name == MAIN_MODEL_NAME:
            found["main"] = model
        elif name == VAE_MODEL_NAME:
            found["vae"] = model
        elif name == ENCODER_MODEL_NAME:
            found["qwen3_encoder"] = model

    missing = [n for role, n in [("main", MAIN_MODEL_NAME), ("vae", VAE_MODEL_NAME), ("qwen3_encoder", ENCODER_MODEL_NAME)] if role not in found]
    if missing:
        raise RuntimeError(f"Required models not found in InvokeAI: {missing}")

    return found


def _model_ref(model, fields=("key", "hash", "name", "base", "type")):
    """Extract a minimal model reference dict for use in graph nodes."""
    return {k: model.get(k) for k in fields}


def build_invokeai_graph(models):
    """
    Build the InvokeAI FLUX graph dynamically using model records
    looked up from the running InvokeAI instance.
    """
    main_model = models["main"]
    vae_model = models["vae"]
    encoder_model = models["qwen3_encoder"]

    # Generate unique node IDs
    graph_id = f"flux_graph:{_random_id()}"
    prompt_id = f"positive_prompt:{_random_id()}"
    seed_id = f"seed:{_random_id()}"
    loader_id = f"flux2_klein_model_loader:{_random_id()}"
    encoder_id = f"flux2_klein_text_encoder:{_random_id()}"
    denoise_id = f"flux2_denoise:{_random_id()}"
    metadata_id = f"core_metadata:{_random_id()}"
    output_id = f"canvas_output:{_random_id()}"

    # Full model record for the loader node
    main_model_full = {
        "key": main_model["key"],
        "hash": main_model["hash"],
        "path": main_model.get("path", ""),
        "file_size": main_model.get("file_size", 0),
        "name": main_model["name"],
        "description": main_model.get("description", ""),
        "source": main_model.get("source", ""),
        "source_type": main_model.get("source_type", "url"),
        "source_api_response": main_model.get("source_api_response"),
        "cover_image": main_model.get("cover_image"),
        "type": "main",
        "trigger_phrases": main_model.get("trigger_phrases"),
        "default_settings": main_model.get("default_settings", {}),
        "config_path": main_model.get("config_path"),
        "base": "flux2",
        "format": main_model.get("format", "gguf_quantized"),
        "variant": main_model.get("variant", "klein_4b"),
    }

    vae_ref = _model_ref(vae_model)
    encoder_ref = _model_ref(encoder_model)

    graph = {
        "id": graph_id,
        "nodes": {
            prompt_id: {
                "id": prompt_id, "type": "string",
                "is_intermediate": True, "use_cache": True
            },
            seed_id: {
                "id": seed_id, "type": "integer",
                "is_intermediate": True, "use_cache": True
            },
            loader_id: {
                "type": "flux2_klein_model_loader", "id": loader_id,
                "model": main_model_full,
                "vae_model": vae_ref,
                "qwen3_encoder_model": encoder_ref,
                "is_intermediate": True, "use_cache": True
            },
            encoder_id: {
                "type": "flux2_klein_text_encoder", "id": encoder_id,
                "is_intermediate": True, "use_cache": True
            },
            denoise_id: {
                "type": "flux2_denoise", "id": denoise_id,
                "num_steps": 30, "is_intermediate": True, "use_cache": True,
                "denoising_start": 0, "denoising_end": 1,
                "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT
            },
            metadata_id: {
                "id": metadata_id, "type": "core_metadata",
                "is_intermediate": True, "use_cache": True,
                "model": _model_ref(main_model),
                "steps": 30,
                "vae": vae_ref,
                "qwen3_encoder": encoder_ref,
                "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
                "generation_mode": "flux2_txt2img"
            },
            output_id: {
                "type": "flux2_vae_decode", "id": output_id,
                "is_intermediate": False, "use_cache": False
            }
        },
        "edges": [
            {"source": {"node_id": loader_id, "field": "qwen3_encoder"}, "destination": {"node_id": encoder_id, "field": "qwen3_encoder"}},
            {"source": {"node_id": loader_id, "field": "max_seq_len"}, "destination": {"node_id": encoder_id, "field": "max_seq_len"}},
            {"source": {"node_id": loader_id, "field": "transformer"}, "destination": {"node_id": denoise_id, "field": "transformer"}},
            {"source": {"node_id": loader_id, "field": "vae"}, "destination": {"node_id": denoise_id, "field": "vae"}},
            {"source": {"node_id": loader_id, "field": "vae"}, "destination": {"node_id": output_id, "field": "vae"}},
            {"source": {"node_id": prompt_id, "field": "value"}, "destination": {"node_id": encoder_id, "field": "prompt"}},
            {"source": {"node_id": encoder_id, "field": "conditioning"}, "destination": {"node_id": denoise_id, "field": "positive_text_conditioning"}},
            {"source": {"node_id": seed_id, "field": "value"}, "destination": {"node_id": denoise_id, "field": "seed"}},
            {"source": {"node_id": denoise_id, "field": "latents"}, "destination": {"node_id": output_id, "field": "latents"}},
            {"source": {"node_id": seed_id, "field": "value"}, "destination": {"node_id": metadata_id, "field": "seed"}},
            {"source": {"node_id": prompt_id, "field": "value"}, "destination": {"node_id": metadata_id, "field": "positive_prompt"}},
            {"source": {"node_id": metadata_id, "field": "metadata"}, "destination": {"node_id": output_id, "field": "metadata"}}
        ]
    }

    # Return graph and the node IDs needed for the data array
    return graph, seed_id, prompt_id


def log(message, level="info"):
    """Print a formatted log message."""
    prefix = {
        "info": "[INFO]", "success": "[OK]", "error": "[ERROR]",
        "warning": "[WARN]", "verbose": "[DEBUG]"
    }
    print(f"  {prefix.get(level, '[INFO]')} {message}")


def backfill_queue(api_url, api_key):
    """Call the backfill endpoint to populate the poster queue."""
    headers = {"X-Api-Key": api_key}
    resp = requests.post(f"{api_url}/poster-queue/backfill", headers=headers)
    resp.raise_for_status()
    return resp.json()


def pop_queue_item(api_url, api_key):
    """Pop the next available item from the poster queue. Returns None if empty."""
    headers = {"X-Api-Key": api_key}
    resp = requests.post(f"{api_url}/poster-queue/pop", headers=headers)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def complete_queue_item(api_url, queue_id, api_key):
    """Mark a queue item as completed."""
    headers = {"X-Api-Key": api_key}
    resp = requests.post(f"{api_url}/poster-queue/{queue_id}/complete", headers=headers)
    resp.raise_for_status()
    return resp.json()


def fail_queue_item(api_url, queue_id, api_key):
    """Report a queue item failure."""
    headers = {"X-Api-Key": api_key}
    resp = requests.post(f"{api_url}/poster-queue/{queue_id}/fail", headers=headers)
    resp.raise_for_status()
    return resp.json()


def build_image_prompt(movie, templates_base, verbose=False):
    """
    Build an image generation prompt using the AI text model.

    Reuses the image_prompt templates from prompts.json and sends them
    to the configured text model (Azure OpenAI or Ollama) to produce
    a detailed image generation prompt for InvokeAI.
    """
    prompt_file_path = os.path.join(templates_base, "prompts.json")
    with open(prompt_file_path) as f:
        prompts_json = json.load(f)

    system_prompt = random.choice(prompts_json["image_prompt_system"])
    # This prompt is always static
    prompt_template = prompts_json["image_prompt"][0]

    # Build replacement values from the movie API response
    replacements = {
        "title": movie.get("title", "Unknown"),
        "tagline": movie.get("tagline", ""),
        "description": movie.get("description", ""),
        "genres": movie.get("genre", ""),
        "mpaa_ratings": movie.get("mpaa_rating", "NR"),
        "eras": "Modern",
    }

    # Fill in template placeholders (same logic as lib/image.py)
    prompt = prompt_template
    start_index = prompt.find("{")
    while start_index != -1:
        end_index = prompt.find("}")
        key = prompt[start_index + 1:end_index]
        key_value = replacements.get(key, "NO VALUE")
        prompt = prompt.replace("{" + key + "}", str(key_value), 1)
        start_index = prompt.find("{")

    if verbose:
        log(f"Image prompt template filled: {prompt}", "verbose")

    # Call the AI text model to generate a detailed image prompt
    model_type = os.getenv("MODEL_TYPE", "").lower()
    if model_type == "azure_openai":
        text_model = aoaiText()
    elif model_type == "local_openai":
        text_model = localOpenAIText()
    else:
        text_model = ollamaText()

    text_model.user_prompt = prompt
    text_model.system_prompt = system_prompt

    completion = text_model.generateResponse()

    # Extract the image_prompt from the JSON response
    start = completion.find("{")
    end = completion.rfind("}") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(completion[start:end])
            image_prompt = result.get("image_prompt", "")
            if image_prompt:
                return image_prompt
        except json.JSONDecodeError:
            pass

    log(f"Failed to parse image prompt from AI response, using template directly", "warning")
    return prompt


def build_invokeai_payload(graph, seed_id, prompt_id, prompt, seed=None):
    """Build the InvokeAI enqueue_batch request body."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "prepend": False,
        "batch": {
            "graph": graph,
            "runs": 1,
            "data": [
                [{"node_path": seed_id, "field_name": "value", "items": [seed]}],
                [{"node_path": prompt_id, "field_name": "value", "items": [prompt]}]
            ],
            "origin": "generate",
            "destination": "generate"
        }
    }


def enqueue_generation(invokeai_url, graph, seed_id, prompt_id, prompt, seed=None):
    """Enqueue an image generation batch in InvokeAI."""
    payload = build_invokeai_payload(graph, seed_id, prompt_id, prompt, seed)
    resp = requests.post(
        f"{invokeai_url}/api/v1/queue/default/enqueue_batch",
        json=payload
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_batch(invokeai_url, batch_id, timeout=GENERATION_TIMEOUT):
    """Poll InvokeAI batch status until complete or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = requests.get(f"{invokeai_url}/api/v1/queue/default/b/{batch_id}/status")
        resp.raise_for_status()
        status = resp.json()

        completed = status.get("completed", 0)
        failed = status.get("failed", 0)
        canceled = status.get("canceled", 0)
        total = status.get("total", 0)

        if failed > 0 or canceled > 0:
            return {"success": False, "status": status}
        if completed >= total and total > 0:
            return {"success": True, "status": status}

        time.sleep(POLL_INTERVAL)

    return {"success": False, "status": {"error": "timeout"}}


def get_latest_image_name(invokeai_url):
    """Get the most recently generated image name from InvokeAI."""
    resp = requests.get(
        f"{invokeai_url}/api/v1/images/",
        params={"limit": 1, "offset": 0}
    )
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if items:
        return items[0].get("image_name")
    return None


def download_image(invokeai_url, image_name):
    """Download a full-resolution image from InvokeAI."""
    resp = requests.get(f"{invokeai_url}/api/v1/images/i/{image_name}/full")
    resp.raise_for_status()
    return BytesIO(resp.content)


def create_thumbnail(image_data: BytesIO) -> BytesIO:
    """Create a 512x896 webp thumbnail from the original image data."""
    image_data.seek(0)
    with Image.open(image_data) as img:
        img = img.convert("RGB")
        img = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
        thumb_buf = BytesIO()
        img.save(thumb_buf, "WEBP", quality=THUMB_QUALITY)
        thumb_buf.seek(0)
    image_data.seek(0)
    return thumb_buf


def upload_poster(api_url, movie_id, image_data, api_key, thumbnail_data=None):
    """Upload a poster image (and optional thumbnail) to the media-generator API."""
    files = [("file", ("poster.png", image_data, "image/png"))]
    if thumbnail_data is not None:
        files.append(("thumbnail", ("poster_thumb.webp", thumbnail_data, "image/webp")))
    headers = {"X-Api-Key": api_key}
    resp = requests.put(f"{api_url}/movies/{movie_id}/poster", files=files, headers=headers)
    resp.raise_for_status()
    return resp.json()


def generate_prompt_for_item(queue_item, templates_base, verbose=False):
    """Generate an image prompt for a queue item. Returns the prompt string or None."""
    movie = queue_item["movie"]
    movie_id = movie["movie_id"]
    title = movie["title"]
    queue_id = queue_item["queue_id"]
    log(f"Generating prompt for movie {movie_id}: '{title}' (queue item {queue_id})")

    try:
        image_prompt = build_image_prompt(movie, templates_base, verbose)
    except Exception as e:
        log(f"  Failed to generate image prompt: {e}", "error")
        if verbose:
            log(traceback.format_exc(), "verbose")
        return None

    if verbose:
        log(f"  Image prompt: {image_prompt}", "verbose")

    log(f"  Prompt generated for '{title}'", "success")
    return image_prompt


def generate_image_for_item(prompt_item, api_url, invokeai_url, api_key, graph, seed_id, prompt_id):
    """Generate an image from a pre-built prompt, upload it, and return success."""
    queue_id = prompt_item["queue_id"]
    movie_id = prompt_item["movie_id"]
    title = prompt_item["title"]
    image_prompt = prompt_item["image_prompt"]
    log(f"Processing image for movie {movie_id}: '{title}' (queue item {queue_id})")

    # Step 1: Enqueue generation in InvokeAI
    log(f"  Enqueuing image generation in InvokeAI...")
    try:
        enqueue_result = enqueue_generation(invokeai_url, graph, seed_id, prompt_id, image_prompt)
        batch_id = enqueue_result.get("batch", {}).get("batch_id")
        if not batch_id:
            log(f"  No batch_id in enqueue response", "error")
            return False
        log(f"  Batch enqueued: {batch_id}")
    except Exception as e:
        log(f"  Failed to enqueue generation: {e}", "error")
        return False

    # Step 2: Wait for generation to complete
    log(f"  Waiting for image generation...")
    result = wait_for_batch(invokeai_url, batch_id)
    if not result["success"]:
        log(f"  Image generation failed: {result['status']}", "error")
        return False
    log(f"  Image generation complete")

    # Step 3: Download the generated image from InvokeAI
    try:
        image_name = get_latest_image_name(invokeai_url)
        if not image_name:
            log(f"  Could not find generated image in InvokeAI", "error")
            return False
        log(f"  Downloading image: {image_name}")
        image_data = download_image(invokeai_url, image_name)
    except Exception as e:
        log(f"  Failed to download image: {e}", "error")
        return False

    # Step 4: Create thumbnail and upload poster to the media-generator API
    try:
        log(f"  Creating thumbnail...")
        thumbnail_data = create_thumbnail(image_data)
        log(f"  Uploading poster and thumbnail for movie {movie_id}...")
        upload_poster(api_url, movie_id, image_data, api_key, thumbnail_data)
        log(f"  Poster uploaded for '{title}'", "success")
        return True
    except Exception as e:
        log(f"  Failed to upload poster: {e}", "error")
        return False


def process_queue_item(queue_item, api_url, invokeai_url, templates_base, api_key, graph, seed_id, prompt_id, verbose=False):
    """Generate and upload a poster for a queue item (combined prompt+image flow)."""
    image_prompt = generate_prompt_for_item(queue_item, templates_base, verbose)
    if image_prompt is None:
        return False

    prompt_item = {
        "queue_id": queue_item["queue_id"],
        "movie_id": queue_item["movie"]["movie_id"],
        "title": queue_item["movie"]["title"],
        "image_prompt": image_prompt,
    }
    return generate_image_for_item(prompt_item, api_url, invokeai_url, api_key, graph, seed_id, prompt_id)


def run_prompts_phase(args, api_key, templates_base):
    """Phase 1: Pop queue items and generate image prompts, saving to a JSON file."""
    prompts_file = args.prompts_file

    print(f"\n=== Batch Poster Generator (Prompts Phase) ===")
    print(f"  Media API:     {args.media_api}")
    print(f"  Prompts file:  {prompts_file}")
    print()

    # Backfill the poster queue
    log("Backfilling poster queue...")
    try:
        backfill_result = backfill_queue(args.media_api, api_key)
        log(f"Queue backfill: {backfill_result.get('added', 0)} added, "
            f"{backfill_result.get('already_queued', 0)} already queued", "success")
    except Exception as e:
        log(f"Failed to backfill queue: {e}", "error")
        return 1

    print()

    prompt_items = []
    fail_count = 0

    # Pop items and generate prompts
    while True:
        try:
            queue_item = pop_queue_item(args.media_api, api_key)
        except Exception as e:
            log(f"Failed to pop from queue: {e}", "error")
            break

        if queue_item is None:
            log("Queue empty, no more items to process.")
            break

        image_prompt = generate_prompt_for_item(queue_item, templates_base, args.verbose)
        if image_prompt is not None:
            prompt_items.append({
                "queue_id": queue_item["queue_id"],
                "movie_id": queue_item["movie"]["movie_id"],
                "title": queue_item["movie"]["title"],
                "image_prompt": image_prompt,
            })
        else:
            queue_id = queue_item["queue_id"]
            try:
                fail_queue_item(args.media_api, queue_id, api_key)
            except Exception:
                pass
            fail_count += 1
        print()

    # Save prompts to JSON file
    os.makedirs(os.path.dirname(prompts_file), exist_ok=True)
    with open(prompts_file, "w") as f:
        json.dump(prompt_items, f, indent=2)

    print(f"=== Prompts phase complete: {len(prompt_items)} prompts saved, {fail_count} failed ===")
    print(f"  Saved to: {prompts_file}\n")
    return 0 if fail_count == 0 else 1


def run_images_phase(args, api_key):
    """Phase 2: Read prompts from JSON file, generate images with InvokeAI, upload."""
    prompts_file = args.prompts_file

    print(f"\n=== Batch Poster Generator (Images Phase) ===")
    print(f"  Media API:     {args.media_api}")
    print(f"  InvokeAI:      {args.invokeai}")
    print(f"  Prompts file:  {prompts_file}")
    print()

    # Load prompts
    if not os.path.exists(prompts_file):
        log(f"Prompts file not found: {prompts_file}", "error")
        return 1

    with open(prompts_file) as f:
        prompt_items = json.load(f)

    if not prompt_items:
        log("No prompts to process.")
        return 0

    log(f"Loaded {len(prompt_items)} prompts from {prompts_file}")

    # Look up model keys from InvokeAI
    log("Looking up models from InvokeAI...")
    try:
        models = lookup_invokeai_models(args.invokeai)
        graph, seed_id, prompt_id = build_invokeai_graph(models)
        log(f"Models found: {', '.join(m['name'] for m in models.values())}", "success")
    except Exception as e:
        log(f"Failed to look up models from InvokeAI: {e}", "error")
        return 1

    print()

    success_count = 0
    fail_count = 0

    for prompt_item in prompt_items:
        queue_id = prompt_item["queue_id"]
        try:
            if generate_image_for_item(
                prompt_item, args.media_api, args.invokeai, api_key,
                graph, seed_id, prompt_id
            ):
                complete_queue_item(args.media_api, queue_id, api_key)
                success_count += 1
            else:
                fail_queue_item(args.media_api, queue_id, api_key)
                fail_count += 1
        except Exception as e:
            log(f"Unexpected error processing queue item {queue_id}: {e}", "error")
            try:
                fail_queue_item(args.media_api, queue_id, api_key)
            except Exception:
                pass
            fail_count += 1
        print()

    # Clean up prompts file after successful processing
    if fail_count == 0:
        try:
            os.remove(prompts_file)
            log(f"Cleaned up prompts file: {prompts_file}")
        except OSError:
            pass

    print(f"=== Images phase complete: {success_count} succeeded, {fail_count} failed ===\n")
    return 0 if fail_count == 0 else 1


def run_all_phase(args, api_key, templates_base):
    """Combined phase: generate prompts and images in one pass (original behavior)."""
    print(f"\n=== Batch Poster Generator ===")
    print(f"  Media API: {args.media_api}")
    print(f"  InvokeAI:  {args.invokeai}")
    print()

    # Look up model keys from InvokeAI
    log("Looking up models from InvokeAI...")
    try:
        models = lookup_invokeai_models(args.invokeai)
        graph, seed_id, prompt_id = build_invokeai_graph(models)
        log(f"Models found: {', '.join(m['name'] for m in models.values())}", "success")
    except Exception as e:
        log(f"Failed to look up models from InvokeAI: {e}", "error")
        return 1

    # Backfill the poster queue with any movies missing posters
    log("Backfilling poster queue...")
    try:
        backfill_result = backfill_queue(args.media_api, api_key)
        log(f"Queue backfill: {backfill_result.get('added', 0)} added, "
            f"{backfill_result.get('already_queued', 0)} already queued", "success")
    except Exception as e:
        log(f"Failed to backfill queue: {e}", "error")
        return 1

    print()

    success_count = 0
    fail_count = 0

    # Pop items from the queue until empty
    while True:
        try:
            queue_item = pop_queue_item(args.media_api, api_key)
        except Exception as e:
            log(f"Failed to pop from queue: {e}", "error")
            break

        if queue_item is None:
            log("Queue empty, no more items to process.")
            break

        queue_id = queue_item["queue_id"]
        try:
            if process_queue_item(
                queue_item, args.media_api, args.invokeai, templates_base,
                api_key, graph, seed_id, prompt_id, args.verbose
            ):
                complete_queue_item(args.media_api, queue_id, api_key)
                success_count += 1
            else:
                fail_queue_item(args.media_api, queue_id, api_key)
                fail_count += 1
        except Exception as e:
            log(f"Unexpected error processing queue item {queue_id}: {e}", "error")
            try:
                fail_queue_item(args.media_api, queue_id, api_key)
            except Exception:
                pass
            fail_count += 1
        print()

    print(f"=== Complete: {success_count} succeeded, {fail_count} failed ===\n")
    return 0 if fail_count == 0 else 1


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Batch generate movie posters for movies missing them."
    )
    parser.add_argument(
        "--phase", choices=["all", "prompts", "images"], default="all",
        help="Which phase to run: 'prompts' generates image prompts and saves to file, "
             "'images' reads prompts and generates images, 'all' does both (default: all)"
    )
    parser.add_argument(
        "--media-api", default=DEFAULT_MEDIA_API_URL,
        help=f"Media generator API URL (default: {DEFAULT_MEDIA_API_URL})"
    )
    parser.add_argument(
        "--invokeai", default=DEFAULT_INVOKEAI_URL,
        help=f"InvokeAI API URL (default: {DEFAULT_INVOKEAI_URL})"
    )
    parser.add_argument(
        "--prompts-file", default=DEFAULT_PROMPTS_FILE,
        help=f"Path to save/load prompts JSON file (default: {DEFAULT_PROMPTS_FILE})"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for the media-generator API (default: from API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("API_KEY", "")
    if not api_key:
        log("No API key provided. Set API_KEY in .env or pass --api-key", "error")
        return 1

    templates_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    if args.phase == "prompts":
        return run_prompts_phase(args, api_key, templates_base)
    elif args.phase == "images":
        return run_images_phase(args, api_key)
    else:
        return run_all_phase(args, api_key, templates_base)


if __name__ == "__main__":
    sys.exit(main())
