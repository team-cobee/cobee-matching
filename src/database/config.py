"""
애플리케이션 설정 파일 - .env 파일 기반
"""
import os
from dotenv import load_dotenv
from typing import Optional
from urllib.parse import quote_plus
# .env 파일 로드
load_dotenv()

class Config:
    """애플리케이션 설정 클래스"""
    # === PostgreSQL 설정 ===
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB_NAME: str = os.getenv("POSTGRES_DB_NAME", "cobee")
    POSTGRES_USERNAME: str = os.getenv("POSTGRES_USERNAME")  # 필수 환경변수
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")  # 필수 환경변수
    
    # === CloudSQL Proxy 설정 ===
    CLOUDSQL_CONNECTION_NAME: str = os.getenv("CLOUDSQL_CONNECTION_NAME","")
    CLOUDSQL_USE_PROXY: bool = os.getenv("CLOUDSQL_USE_PROXY","false").lower() == "true"
    POSTGRES_SSL_MODE: str = os.getenv("POSTGRES_SSL_MODE", "prefer")
    POSTGRES_USE_IAM: bool = os.getenv("POSTGRES_USE_IAM", "false").lower() == "true"
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")    

    # === MongoDB 설정 ===
    MONGO_ROOT_USERNAME: str = os.getenv("MONGO_ROOT_USERNAME")  # 필수 환경변수
    MONGO_ROOT_PASSWORD: str = os.getenv("MONGO_ROOT_PASSWORD")  # 필수 환경변수
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "roommate_matching")
    MONGODB_APP_USER: str = os.getenv("MONGODB_APP_USER")  # 필수 환경변수
    MONGODB_APP_PASSWORD: str = os.getenv("MONGODB_APP_PASSWORD")  # 필수 환경변수
    
    # === ML 설정 ===
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models/baseline")
    POLLING_INTERVAL: int = int(os.getenv("POLLING_INTERVAL_SECONDS", "30"))
    FEATURE_CACHE_TTL: int = int(os.getenv("FEATURE_CACHE_TTL", "300"))
    RECOMMENDATION_CACHE_TTL: int = int(os.getenv("RECOMMENDATION_CACHE_TTL", "60"))
    
    # === 애플리케이션 설정 ===
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # === 로깅 설정 ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

# 필수 환경변수 검증
def validate_required_env_vars():
    """필수 환경변수들이 설정되어 있는지 검증"""
    required_vars = [
        "POSTGRES_USERNAME", "POSTGRES_PASSWORD",
        "MONGO_ROOT_USERNAME", "MONGO_ROOT_PASSWORD", 
        "MONGODB_APP_USER", "MONGODB_APP_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")

# 환경변수 검증 실행
validate_required_env_vars()

# 전역 설정 인스턴스
config = Config()

# 편의 함수들
def get_postgres_config() -> dict:
    """PostgreSQL 연결 설정 반환"""
    # CloudSQL Proxy 사용 시 특별한 설정
    if config.CLOUDSQL_USE_PROXY and config.CLOUDSQL_CONNECTION_NAME:
        return {
            "connection_name": config.CLOUDSQL_CONNECTION_NAME,
            "database": config.POSTGRES_DB_NAME,
            "user": config.POSTGRES_USERNAME,
            "password": config.POSTGRES_PASSWORD,
            "use_proxy": True,
            "use_iam": config.POSTGRES_USE_IAM,
            "credentials_path": config.GOOGLE_APPLICATION_CREDENTIALS
        }
    else:
        return{
            "host": config.POSTGRES_HOST,
            "port": config.POSTGRES_PORT,
            "database": config.POSTGRES_DB_NAME,
            "user": config.POSTGRES_USERNAME,
            "password": config.POSTGRES_PASSWORD
        }
    
def get_mongodb_config() -> dict:
    """MongoDB 연결 설정 반환"""
    # MONGODB_URL 환경 변수가 있으면 사용 (Atlas 또는 커스텀)
    mongodb_url = os.getenv("MONGODB_URL")

    if mongodb_url:
        # 환경 변수 치환
        username = quote_plus(config.MONGODB_APP_USER)
        password = quote_plus(config.MONGODB_APP_PASSWORD)
        db_name = config.MONGODB_DB_NAME

        mongodb_url = mongodb_url.replace("${MONGODB_APP_USER}",username)
        mongodb_url = mongodb_url.replace("${MONGODB_APP_PASSWORD}", password)
        mongodb_url = mongodb_url.replace("${MONGODB_DB_NAME}", db_name)
    else:
        username = quote_plus(config.MONGO_ROOT_USERNAME)
        password = quote_plus(config.MONGO_ROOT_PASSWORD)
        mongodb_url = f"mongodb://{username}:{password}@localhost:27018/{config.MONGODB_DB_NAME}?authSource=admin"
    return {
        "url": mongodb_url,
        "database": config.MONGODB_DB_NAME
    }