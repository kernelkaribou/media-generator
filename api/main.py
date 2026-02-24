"""
FastAPI Web API for the media-generator library.

Provides REST endpoints for generating AI-powered media objects
and storing them in a SQL Server database.

Usage:
    uvicorn api.main:app --reload
    
Or run directly:
    python -m api.main
"""

import random
import sys
import os
import shutil
from contextlib import asynccontextmanager
from datetime import date
from enum import Enum
from typing import Optional, List
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Depends, Query, Request, UploadFile, File, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import joinedload
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from lib import MediaGenerator
from api.models import (
    MovieModel, GenreModel, ActorModel, DirectorModel, CriticReviewModel,
    PosterQueueModel,
    create_db_engine, get_session_factory,
    get_or_create_genre, get_or_create_actor, get_or_create_director
)


# Pydantic models for API request/response
class SortBy(str, Enum):
    """Sortable fields for movie listings."""
    movie_id = "movie_id"
    title = "title"
    release_date = "release_date"
    popularity_score = "popularity_score"


class SortOrder(str, Enum):
    """Sort direction."""
    asc = "asc"
    desc = "desc"


class GenerateRequest(BaseModel):
    """Request model for generating media."""
    count: int = Field(default=1, ge=1, le=5, description="Number of media objects to generate (1-5)")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class ReviewResponse(BaseModel):
    """Response model for a critic review."""
    critic_review_id: Optional[int] = None
    critic_review: Optional[str] = None
    critic_score: Optional[float] = None

    class Config:
        from_attributes = True


class ActorResponse(BaseModel):
    """Response model for an actor."""
    actor_id: int
    actor: str
    image_url: Optional[str] = None

    class Config:
        from_attributes = True


class DirectorResponse(BaseModel):
    """Response model for a director."""
    director_id: int
    director: str
    image_url: Optional[str] = None

    class Config:
        from_attributes = True


class MovieResponse(BaseModel):
    """Response model for a movie."""
    movie_id: Optional[int] = None
    external_id: str
    title: str
    tagline: Optional[str] = None
    mpaa_rating: Optional[str] = None
    description: Optional[str] = None
    popularity_score: Optional[float] = None
    genre: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[date] = None
    actors: List[ActorResponse] = []
    directors: List[DirectorResponse] = []
    reviews: List[ReviewResponse] = []

    class Config:
        from_attributes = True


class GenerateResponse(BaseModel):
    """Response model for generation results."""
    success: bool
    message: str
    generated_count: int
    movies: List[MovieResponse] = []


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    total_movies: int
    total_reviews: int
    total_actors: int
    total_directors: int
    genres: dict
    ratings: dict


class AppState:
    """Application state container for database connection."""
    engine = None
    session_factory = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler - setup and teardown."""
    # Startup
    print("Initializing database connection...")
    AppState.engine = create_db_engine(echo=False)
    # Don't create tables - using existing schema
    AppState.session_factory = get_session_factory(AppState.engine)
    print("Database initialized successfully.")
    
    yield
    
    # Shutdown
    if AppState.engine:
        AppState.engine.dispose()
        print("Database connection closed.")


app = FastAPI(
    title="Media Generator API",
    description="Generate AI-powered fake media objects including movie titles, descriptions, critic reviews, and poster images.",
    version="1.0.0",
    lifespan=lifespan
)


# Rate limiting setup - use X-Real-IP header if available, fallback to remote address
def get_real_ip(request: Request) -> str:
    """Get client IP from X-Real-IP header or fallback to direct connection."""
    return request.headers.get("X-Real-IP", get_remote_address(request))


limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def get_db() -> Session:
    """Dependency to get database session."""
    if AppState.session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = AppState.session_factory()  # pylint: disable=not-callable
    try:
        yield db
    finally:
        db.close()


def verify_api_key(x_api_key: str = Header(..., description="API key for write operations")):
    """Dependency to validate API key from the X-Api-Key header."""
    valid_keys = os.getenv("API_KEYS", "").split(",")
    valid_keys = [k.strip() for k in valid_keys if k.strip()]
    if not valid_keys:
        raise HTTPException(status_code=500, detail="No API keys configured on server")
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")


def generate_random_release_date() -> date:
    """Generate a random release date within the last 50 years."""
    today = date.today()
    days_back = random.randint(0, 365 * 50)  # Up to 50 years back
    return date.fromordinal(today.toordinal() - days_back)


def save_movie_to_db(media_object, db: Session) -> MovieModel:
    """Save a generated media object to the database using the existing schema."""
    
    # Get or create genre
    genre_name = media_object.genre or "Unknown"
    genre = get_or_create_genre(db, genre_name)
    
    # Create the movie record
    movie_record = MovieModel(
        external_id=media_object.media_id[:30],  # Truncate to fit schema
        title=media_object.title[:100],  # Truncate to fit schema
        tagline=(media_object.tagline or "")[:500],
        description=(media_object.description or "")[:2000],
        mpaa_rating=(media_object.mpaa_rating or "NR")[:5],
        popularity_score=Decimal(str(media_object.popularity_score)) if media_object.popularity_score else None,
        genre_id=genre.genre_id,
        poster_url=getattr(media_object, 'poster_url', None),
        release_date=date.today(),
    )
    
    db.add(movie_record)
    db.flush()  # Get the movie_id
    
    # Add actors from the prompt list
    actors_list = media_object.object_prompt_list.get("actors", [])
    for actor_name in actors_list:
        if actor_name:
            actor = get_or_create_actor(db, actor_name[:500])
            if actor not in movie_record.actors:
                movie_record.actors.append(actor)
    
    # Add directors from the prompt list
    directors_list = media_object.object_prompt_list.get("directors", [])
    for director_name in directors_list:
        if director_name:
            director = get_or_create_director(db, director_name[:500])
            if director not in movie_record.directors:
                movie_record.directors.append(director)
    
    # Add critic reviews
    for review_data in media_object.reviews:
        review_text = review_data.get("review", "")[:4000]
        review_record = CriticReviewModel(
            movie_id=movie_record.movie_id,
            critic_score=Decimal(str(review_data.get("score", 0))) if review_data.get("score") else None,
            critic_review=review_text or "No review provided",
        )
        db.add(review_record)
    
    # Auto-enqueue for poster generation
    enqueue_movie(db, movie_record.movie_id)
    
    db.commit()
    db.refresh(movie_record)
    
    return movie_record


def movie_to_response(movie: MovieModel) -> MovieResponse:
    """Convert a MovieModel to a MovieResponse."""
    return MovieResponse(
        movie_id=movie.movie_id,
        external_id=movie.external_id,
        title=movie.title,
        tagline=movie.tagline,
        mpaa_rating=movie.mpaa_rating,
        description=movie.description,
        popularity_score=float(movie.popularity_score) if movie.popularity_score else None,
        genre=movie.genre_rel.genre if movie.genre_rel else None,
        poster_url=movie.poster_url,
        release_date=movie.release_date,
        actors=[ActorResponse(actor_id=a.actor_id, actor=a.actor, image_url=f"/images/actors/{a.actor_id}.png") for a in movie.actors],
        directors=[DirectorResponse(director_id=d.director_id, director=d.director, image_url=f"/images/directors/{d.director_id}.png") for d in movie.directors],
        reviews=[
            ReviewResponse(
                critic_review_id=r.critic_review_id,
                critic_review=r.critic_review,
                critic_score=float(r.critic_score) if r.critic_score else None
            ) for r in movie.reviews
        ]
    )


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "media-generator-api", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check including database connectivity."""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except SQLAlchemyError as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status,
        "model_type": os.getenv("MODEL_TYPE", "not set")
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
@limiter.limit("10/minute")
async def generate_media(
    request: Request,
    generate_request: GenerateRequest,
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Generate new media objects.
    
    Generates AI-powered fake movie/media information and saves to the database.
    Returns the generated media objects as JSON.
    
    Rate limited to 10 requests per minute per IP.
    """
    try:
        generator = MediaGenerator(
            verbose=generate_request.verbose,
            dry_run=True,  # Don't save files, we save to DB instead
            skip_image=True  # Posters are generated via the batch process
        )
        
        generated_movies = []
        success_count = 0
        
        for _ in range(generate_request.count):
            result = generator.generate_single(save=False)
            
            if result.success and result.media_object:
                # Save to database (also enqueues for poster generation)
                movie_record = save_movie_to_db(result.media_object, db)
                
                # Build response
                movie_response = movie_to_response(movie_record)
                generated_movies.append(movie_response)
                success_count += 1
        
        return GenerateResponse(
            success=success_count > 0,
            message=f"Generated {success_count} of {generate_request.count} movies",
            generated_count=success_count,
            movies=generated_movies
        )
        
    except (SQLAlchemyError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}") from e


@app.get("/movies", response_model=List[MovieResponse], tags=["Movies"])
async def list_movies(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    sort_by: SortBy = Query(SortBy.movie_id, description="Field to sort by"),
    order: SortOrder = Query(SortOrder.desc, description="Sort direction"),
    start_id: Optional[int] = Query(None, ge=1, description="Return movies starting at this movie_id"),
    db: Session = Depends(get_db)
):
    """
    List all movies from the database.
    
    Supports pagination, filtering by genre, sorting, and cursor-based
    pagination via start_id (filters to movie_id >= start_id regardless
    of sort order).
    """
    query = db.query(MovieModel)
    
    if start_id is not None:
        query = query.filter(MovieModel.movie_id >= start_id)
    
    if genre:
        query = query.join(GenreModel).filter(GenreModel.genre.ilike(f"%{genre}%"))
    
    sort_column = getattr(MovieModel, sort_by.value)
    if order == SortOrder.asc:
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    movies = query.offset(skip).limit(limit).all()
    
    return [movie_to_response(m) for m in movies]


@app.get("/movies/random", response_model=List[MovieResponse], tags=["Movies"])
async def get_random_movies(db: Session = Depends(get_db)):
    """
    Get 3 random movies.
    """
    random_movies = db.query(MovieModel).order_by(func.newid()).limit(3).all()  # pylint: disable=not-callable
    
    return [movie_to_response(m) for m in random_movies]


@app.get("/genres", response_model=List[dict], tags=["Lookup"])
async def list_genres(db: Session = Depends(get_db)):
    """List all genres."""
    genres = db.query(GenreModel).order_by(GenreModel.genre).all()
    return [{"genre_id": g.genre_id, "genre": g.genre} for g in genres]


@app.get("/actors", response_model=List[ActorResponse], tags=["Lookup"])
async def list_actors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """List all actors with pagination."""
    actors = db.query(ActorModel).order_by(ActorModel.actor).offset(skip).limit(limit).all()
    return [ActorResponse(actor_id=a.actor_id, actor=a.actor, image_url=f"/images/actors/{a.actor_id}.png") for a in actors]


@app.get("/directors", response_model=List[DirectorResponse], tags=["Lookup"])
async def list_directors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """List all directors with pagination."""
    directors = db.query(DirectorModel).order_by(DirectorModel.director).offset(skip).limit(limit).all()
    return [DirectorResponse(director_id=d.director_id, director=d.director, image_url=f"/images/directors/{d.director_id}.png") for d in directors]


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the movie database.
    """
    total_movies = db.query(func.count(MovieModel.movie_id)).scalar() or 0  # pylint: disable=not-callable
    total_reviews = db.query(func.count(CriticReviewModel.critic_review_id)).scalar() or 0  # pylint: disable=not-callable
    total_actors = db.query(func.count(ActorModel.actor_id)).scalar() or 0  # pylint: disable=not-callable
    total_directors = db.query(func.count(DirectorModel.director_id)).scalar() or 0  # pylint: disable=not-callable
    
    # Genre distribution
    genre_counts = db.query(
        GenreModel.genre, 
        func.count(MovieModel.movie_id)  # pylint: disable=not-callable
    ).join(MovieModel).group_by(GenreModel.genre).all()
    genres = {g[0]: g[1] for g in genre_counts}
    
    # Rating distribution
    rating_counts = db.query(
        MovieModel.mpaa_rating,
        func.count(MovieModel.movie_id)  # pylint: disable=not-callable
    ).group_by(MovieModel.mpaa_rating).all()
    ratings = {r[0] or "NR": r[1] for r in rating_counts}
    
    return StatsResponse(
        total_movies=total_movies,
        total_reviews=total_reviews,
        total_actors=total_actors,
        total_directors=total_directors,
        genres=genres,
        ratings=ratings
    )


@app.get("/movies/top-rated", response_model=List[MovieResponse], tags=["Movies"])
async def get_top_rated_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 highest rated movies by average critic score.
    """
    # Subquery to get average critic score per movie
    avg_scores = db.query(
        CriticReviewModel.movie_id,
        func.avg(CriticReviewModel.critic_score).label("avg_score")  # pylint: disable=not-callable
    ).group_by(CriticReviewModel.movie_id).subquery()
    
    # Join with movies and order by average score descending
    top_movies = db.query(MovieModel).join(
        avg_scores, MovieModel.movie_id == avg_scores.c.movie_id
    ).order_by(avg_scores.c.avg_score.desc()).limit(5).all()
    
    return [movie_to_response(m) for m in top_movies]


@app.get("/movies/worst-rated", response_model=List[MovieResponse], tags=["Movies"])
async def get_worst_rated_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 lowest rated movies by average critic score.
    """
    # Subquery to get average critic score per movie
    avg_scores = db.query(
        CriticReviewModel.movie_id,
        func.avg(CriticReviewModel.critic_score).label("avg_score")  # pylint: disable=not-callable
    ).group_by(CriticReviewModel.movie_id).subquery()
    
    # Join with movies and order by average score ascending
    worst_movies = db.query(MovieModel).join(
        avg_scores, MovieModel.movie_id == avg_scores.c.movie_id
    ).order_by(avg_scores.c.avg_score.asc()).limit(5).all()
    
    return [movie_to_response(m) for m in worst_movies]


@app.get("/movies/recent", response_model=List[MovieResponse], tags=["Movies"])
async def get_recent_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 most recently released movies.
    """
    recent_movies = db.query(MovieModel).order_by(
        MovieModel.release_date.desc()
    ).limit(5).all()
    
    return [movie_to_response(m) for m in recent_movies]


@app.get("/genres/top", response_model=List[dict], tags=["Lookup"])
async def get_top_genres(db: Session = Depends(get_db)):
    """
    Get the top 5 genres by movie count.
    """
    top_genres = db.query(
        GenreModel.genre_id,
        GenreModel.genre,
        func.count(MovieModel.movie_id).label("movie_count")  # pylint: disable=not-callable
    ).join(MovieModel).group_by(
        GenreModel.genre_id, GenreModel.genre
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()  # pylint: disable=not-callable
    
    return [{"genre_id": g.genre_id, "genre": g.genre, "movie_count": g.movie_count} for g in top_genres]


@app.get("/actors/top", response_model=List[dict], tags=["Lookup"])
async def get_top_actors(db: Session = Depends(get_db)):
    """
    Get the top 5 actors by movie count.
    """
    top_actors = db.query(
        ActorModel.actor_id,
        ActorModel.actor,
        func.count(MovieModel.movie_id).label("movie_count")  # pylint: disable=not-callable
    ).join(ActorModel.movies).group_by(
        ActorModel.actor_id, ActorModel.actor
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()  # pylint: disable=not-callable
    
    return [{"actor_id": a.actor_id, "actor": a.actor, "movie_count": a.movie_count, "image_url": f"/images/actors/{a.actor_id}.png"} for a in top_actors]


@app.get("/directors/top", response_model=List[dict], tags=["Lookup"])
async def get_top_directors(db: Session = Depends(get_db)):
    """
    Get the top 5 directors by movie count.
    """
    top_directors = db.query(
        DirectorModel.director_id,
        DirectorModel.director,
        func.count(MovieModel.movie_id).label("movie_count")  # pylint: disable=not-callable
    ).join(DirectorModel.movies).group_by(
        DirectorModel.director_id, DirectorModel.director
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()  # pylint: disable=not-callable
    
    return [{"director_id": d.director_id, "director": d.director, "movie_count": d.movie_count, "image_url": f"/images/directors/{d.director_id}.png"} for d in top_directors]


@app.get("/actors/{actor_id}", response_model=ActorResponse, tags=["Lookup"])
async def get_actor(actor_id: int, db: Session = Depends(get_db)):
    """Get a single actor by ID."""
    actor = db.query(ActorModel).filter(ActorModel.actor_id == actor_id).first()
    if not actor:
        raise HTTPException(status_code=404, detail="Actor not found")
    return ActorResponse(actor_id=actor.actor_id, actor=actor.actor, image_url=f"/images/actors/{actor.actor_id}.png")


@app.get("/actors/{actor_id}/movies", response_model=List[MovieResponse], tags=["Movies"])
async def get_movies_by_actor(actor_id: int, db: Session = Depends(get_db)):
    """
    Get all movies featuring a specific actor.
    """
    actor = db.query(ActorModel).filter(ActorModel.actor_id == actor_id).first()
    if not actor:
        raise HTTPException(status_code=404, detail="Actor not found")

    return [movie_to_response(m) for m in actor.movies]


@app.get("/directors/{director_id}", response_model=DirectorResponse, tags=["Lookup"])
async def get_director(director_id: int, db: Session = Depends(get_db)):
    """Get a single director by ID."""
    director = db.query(DirectorModel).filter(DirectorModel.director_id == director_id).first()
    if not director:
        raise HTTPException(status_code=404, detail="Director not found")
    return DirectorResponse(director_id=director.director_id, director=director.director, image_url=f"/images/directors/{director.director_id}.png")


@app.get("/directors/{director_id}/movies", response_model=List[MovieResponse], tags=["Movies"])
async def get_movies_by_director(director_id: int, db: Session = Depends(get_db)):
    """
    Get all movies by a specific director.
    """
    director = db.query(DirectorModel).filter(DirectorModel.director_id == director_id).first()
    if not director:
        raise HTTPException(status_code=404, detail="Director not found")

    return [movie_to_response(m) for m in director.movies]


@app.get("/genres/{genre_id}/movies", response_model=List[MovieResponse], tags=["Movies"])
async def get_movies_by_genre(genre_id: int, db: Session = Depends(get_db)):
    """
    Get all movies in a specific genre.
    """
    genre = db.query(GenreModel).filter(GenreModel.genre_id == genre_id).first()
    if not genre:
        raise HTTPException(status_code=404, detail="Genre not found")

    return [movie_to_response(m) for m in genre.movies]


@app.get("/movies/missing-posters", response_model=List[MovieResponse], tags=["Movies"])
async def get_movies_missing_posters(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """
    Get all movies that are missing poster images.

    Returns movies where poster_url is NULL or set to the placeholder value.
    """
    movies = db.query(MovieModel).filter(
        (MovieModel.poster_url.is_(None)) | (MovieModel.poster_url == "movie_poster_url.jpeg")
    ).order_by(MovieModel.movie_id.asc()).offset(skip).limit(limit).all()

    return [movie_to_response(m) for m in movies]


@app.put("/movies/{movie_id}/poster", response_model=MovieResponse, tags=["Movies"])
async def upload_movie_poster(
    movie_id: int,
    file: UploadFile = File(..., description="Poster image file"),
    thumbnail: Optional[UploadFile] = File(None, description="Optional webp thumbnail image"),
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Upload a poster image for a movie and update its poster_url.

    Accepts an image file upload, saves it to the images directory,
    and updates the movie's poster_url in the database.
    If a thumbnail is provided, the poster_url will point to the thumbnail instead.
    """
    movie = db.query(MovieModel).filter(MovieModel.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    valid_types = ("image/jpeg", "image/png", "image/webp")
    ext_map = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

    # Validate and save the original image
    if file.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="File must be a JPEG, PNG, or WebP image")

    ext = ext_map[file.content_type]
    filename = f"movie_{movie_id}.{ext}"

    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(img_dir, exist_ok=True)
    file_path = os.path.join(img_dir, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}") from e

    # Default poster_url points to the original
    poster_url = f"/images/{filename}"

    # Save thumbnail if provided
    if thumbnail is not None:
        if thumbnail.content_type not in valid_types:
            raise HTTPException(
                status_code=400, detail="Thumbnail must be a JPEG, PNG, or WebP image"
            )
        thumb_ext = ext_map[thumbnail.content_type]
        thumb_filename = f"movie_{movie_id}_thumb.{thumb_ext}"
        thumb_path = os.path.join(img_dir, thumb_filename)

        try:
            with open(thumb_path, "wb") as buffer:
                shutil.copyfileobj(thumbnail.file, buffer)
        except IOError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save thumbnail: {str(e)}"
            ) from e

        poster_url = f"/images/{thumb_filename}"

    movie.poster_url = poster_url
    db.commit()
    db.refresh(movie)

    return movie_to_response(movie)


@app.get("/movies/{movie_id}", response_model=MovieResponse, tags=["Movies"])
async def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """
    Get a specific movie by ID.
    """
    movie = db.query(MovieModel).filter(MovieModel.movie_id == movie_id).first()
    
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return movie_to_response(movie)


# --- Poster Queue Endpoints ---

STALE_CLAIM_MINUTES = 10


class PosterQueueResponse(BaseModel):
    """Response model for a poster queue item."""
    queue_id: int
    movie_id: int
    status: str
    attempt_count: int
    max_attempts: int
    movie: Optional[MovieResponse] = None

    class Config:
        from_attributes = True


class PosterQueueStatsResponse(BaseModel):
    """Response model for queue statistics."""
    pending: int
    claimed: int
    completed: int
    failed: int
    total: int


def queue_item_to_response(item: PosterQueueModel, include_movie: bool = False) -> dict:
    """Convert a PosterQueueModel to a response dict."""
    resp = {
        "queue_id": item.queue_id,
        "movie_id": item.movie_id,
        "status": item.status,
        "attempt_count": item.attempt_count,
        "max_attempts": item.max_attempts,
    }
    if include_movie and item.movie:
        resp["movie"] = movie_to_response(item.movie)
    return resp


def enqueue_movie(db: Session, movie_id: int) -> Optional[PosterQueueModel]:
    """Add a movie to the poster queue if not already present."""
    existing = db.query(PosterQueueModel).filter(
        PosterQueueModel.movie_id == movie_id
    ).first()
    if existing:
        return None
    item = PosterQueueModel(movie_id=movie_id, status="pending")
    db.add(item)
    return item


@app.post("/poster-queue/backfill", response_model=dict, tags=["Poster Queue"])
async def backfill_poster_queue(
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Scan for movies missing posters and add them to the queue.

    First removes failed queue items older than 5 minutes so their movies
    can be re-enqueued. Then finds movies where poster_url is NULL or the
    placeholder value and adds them to the poster queue if not already present.
    """
    from datetime import datetime, timedelta

    failed_cutoff = datetime.utcnow() - timedelta(minutes=5)
    removed = db.query(PosterQueueModel).filter(
        PosterQueueModel.status == "failed",
        PosterQueueModel.claimed_at < failed_cutoff,
    ).delete()

    movies = db.query(MovieModel.movie_id).filter(
        (MovieModel.poster_url.is_(None)) | (MovieModel.poster_url == "movie_poster_url.jpeg")
    ).all()

    added = 0
    for (movie_id,) in movies:
        if enqueue_movie(db, movie_id):
            added += 1

    try:
        db.commit()
    except IntegrityError:
        # Concurrent backfill created duplicates; roll back and count what actually got added
        db.rollback()
        added = 0
        removed = 0

    return {"added": added, "already_queued": len(movies) - added, "removed_failed": removed}


@app.post("/poster-queue/pop", response_model=Optional[PosterQueueResponse], tags=["Poster Queue"])
async def pop_poster_queue(
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Atomically claim the next available item from the poster queue.

    Returns the queue item with full movie data, or null if the queue is empty.
    Items that have been claimed for longer than 10 minutes are considered stale
    and will be reclaimed.
    """
    from datetime import datetime, timedelta

    stale_cutoff = datetime.utcnow() - timedelta(minutes=STALE_CLAIM_MINUTES)

    # Use with_for_update to lock the row for atomic claim
    item = db.query(PosterQueueModel).options(
        joinedload(PosterQueueModel.movie)
    ).filter(
        ((PosterQueueModel.status == "pending") |
         ((PosterQueueModel.status == "claimed") & (PosterQueueModel.claimed_at < stale_cutoff)))
    ).order_by(PosterQueueModel.queue_id).with_for_update(skip_locked=True).first()

    if not item:
        return JSONResponse(content=None, status_code=204)

    item.status = "claimed"
    item.claimed_at = datetime.utcnow()
    item.attempt_count += 1
    db.commit()
    db.refresh(item)

    return queue_item_to_response(item, include_movie=True)


@app.post(
    "/poster-queue/{queue_id}/complete",
    response_model=PosterQueueResponse,
    tags=["Poster Queue"],
)
async def complete_poster_queue_item(
    queue_id: int,
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Mark a poster queue item as completed.
    """
    from datetime import datetime

    item = db.query(PosterQueueModel).filter(
        PosterQueueModel.queue_id == queue_id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    item.status = "completed"
    item.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(item)

    return queue_item_to_response(item)


@app.post(
    "/poster-queue/{queue_id}/fail",
    response_model=PosterQueueResponse,
    tags=["Poster Queue"],
)
async def fail_poster_queue_item(
    queue_id: int,
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Report a poster generation failure.

    If the item has remaining attempts, it goes back to pending.
    Otherwise it is marked as failed permanently.
    """
    item = db.query(PosterQueueModel).filter(
        PosterQueueModel.queue_id == queue_id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    if item.attempt_count >= item.max_attempts:
        item.status = "failed"
    else:
        item.status = "pending"
        item.claimed_at = None

    db.commit()
    db.refresh(item)

    return queue_item_to_response(item)


@app.get("/poster-queue/stats", response_model=PosterQueueStatsResponse, tags=["Poster Queue"])
async def poster_queue_stats(
    _api_key: None = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get poster queue statistics.
    """
    counts = {}
    for status in ("pending", "claimed", "completed", "failed"):
        counts[status] = db.query(func.count(PosterQueueModel.queue_id)).filter(  # pylint: disable=not-callable
            PosterQueueModel.status == status
        ).scalar() or 0

    return PosterQueueStatsResponse(
        pending=counts["pending"],
        claimed=counts["claimed"],
        completed=counts["completed"],
        failed=counts["failed"],
        total=sum(counts.values()),
    )

# Mount static files for image hosting
images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
if os.path.exists(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
