import psycopg2
import psycopg2.extras # RealDictCursor : 결과가 딕셔너리 {'id': 1, 'name': 'John', 'gender': 'Male'}로 나옴 -> 컬럼명으로 접근 가능해 편리
from psycopg2.pool import SimpleConnectionPool  # 연결 풀
import logging
from typing import Optional, Dict, List
from datetime import datetime
from .config import get_postgres_config
from .mongodb_manager import get_mongodb_manager

from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import os

logger = logging.getLogger(__name__)

class PostgreSQLClient:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """PostgreSQL 클라이언트 초기화 (CloudSQL Proxy 지원)"""
        # config에서 전체 설정 가져오기
        postgres_config = get_postgres_config()

        if postgres_config.get("use_proxy", False):
            self._init_cloudsql_proxy(postgres_config)
        else:
            # 기존 연결 방식
            self.connection_params = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database
            }
            self.connection_pool = None
            self.use_proxy = False
            self._initialize_connection_pool()

    def _init_cloudsql_proxy(self, config: dict):
        """Cloud SQL Proxy 연결 초기화"""
        try:
            # 서비스 계정 키 파일 설정
            credentials_path = config.get("credentials_path")
            if credentials_path and os.path.exists(credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.connector = Connector()
            self.use_proxy = True
            
            def getconn():
                conn = self.connector.connect(
                    config["connection_name"],
                    "pg8000",
                    user=config["user"],
                    password=config["password"],
                    db=config["database"],
                    enable_iam_auth=config.get("use_iam", False)
                )
                return conn
            # SQLAlchemy 엔진 생성
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn,
                poolclass=NullPool, #Proxy가 연결 관리
            )
            logger.info("Cloud SQL Proxy 연결 초기화 완료")
        except Exception as e:
            logger.error(f"Cloud SQL Proxy 초기화 실패 : {e}")
            raise
    def _initialize_connection_pool(self) -> None:
        """PostgreSQL 연결 풀 초기화"""
        try:
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **self.connection_params
            )
            if self.connection_pool:
                logging.info("PostgreSQL 연결 풀 초기화 완료")
        except Exception as e:
            logging.error(f"PostgreSQL 연결 풀 초기화 실패: {e}")
            self.connection_pool = None
    
    def _get_connection(self):
        """연결 가져오기 (Proxy 또는 기존 방식)"""
        if hasattr(self, 'use_proxy') and self.use_proxy:
            return self.engine.connect()
        else:
            # 기존 연결 풀 방식
            if not self.connection_pool:
                raise Exception("PostgreSQL 연결 풀이 초기화되지 않음")
            return self.connection_pool.getconn()    
    
    def _return_connection(self, conn):
        """연결 반환 (Proxy 또는 기존 방식)"""
        if hasattr(self, 'use_proxy') and self.use_proxy:
            conn.close()
        else:
            if self.connection_pool and conn:
                self.connection_pool.putconn(conn)

    def health_check(self) -> dict:
        """데이터베이스 연결 상태 확인 (Proxy 지원)"""
        try:
            conn = self._get_connection()

            if hasattr(self, 'use_proxy') and self.use_proxy:
                #SQLAlchemy 연결 사용
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                is_connected = row is not None and row[0] == 1
                connection_info = {"type": "cloudsql_proxy", "connection_name": "masked"}
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                cursor.close()
                is_connected = result is not None and result[0] == 1
                connection_info = self.connection_params

            self._return_connection(conn)
            
            return {
                "is_connected": is_connected,
                "connection_info": connection_info,
                "checked_at": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"PostgreSQL 헬스체크 실패: {e}")
            return {
                "is_connected": False,
                "connection_info": getattr(self, 'connection_params', {}),
                "error": str(e),
                "checked_at": datetime.now().isoformat()
            }

    def disconnect(self) -> None:
        """PostgreSQL 연결 종료"""
        try:
            if hasattr(self, 'use_proxy') and self.use_proxy:
                # CloudSQL Proxy 연결 종료
                if hasattr(self, 'connector'):
                    self.connector.close()
                logging.info("CloudSQL Proxy 연결 종료 완료")
            else:
                # 기존 연결 풀 종료
                if hasattr(self, 'connection_pool') and self.connection_pool:
                    self.connection_pool.closeall()
                    self.connection_pool = None
                logging.info("PostgreSQL 연결 종료 완료")
        except Exception as e:
            logging.error(f"PostgreSQL 연결 종료 실패: {e}")

    def check_data_changes(self) -> dict: # 호출 관계: polling_scheduler._check_data_changes() -> 에서 이 메서드 호출
        """
        데이터 변경분 확인 (폴링 스케줄러에서 호출)
        """
        try:
            # 직전 폴링 시점 이후 변경된 데이터가 있는지 확인 (mongodb polling_status 컬렉션의 last_data_change 참고)
            # 1. MongoDB에서 last_data_change 시간 가져오기
            mongodb_manager = get_mongodb_manager()
            last_check_time = mongodb_manager.get_last_check_time()

            # 2. PostgreSQL에서 해당 시간 이후로 변경된 데이터가 있는지 확인
            query = """
            SELECT 'member' as table_name, COUNT(*) AS change_count
            FROM member
            WHERE updated_at > %s
            UNION ALL
            SELECT 'recruit_post' as table_name, COUNT(*) AS change_count
            FROM recruit_post
            WHERE updated_at > %s
            UNION ALL
            SELECT 'bookmark' as table_name, COUNT(*) AS change_count
            FROM bookmark
            WHERE updated_at > %s
            """
            # execute_query 사용으로 변경
            changes = self.execute_query(query, (last_check_time,last_check_time, last_check_time))
            # 3. 변경분 계산
            total_changes = sum(row['change_count'] for row in changes)
            has_changes = total_changes > 0
            changes_by_table = {row['table_name']: row['change_count'] for row in changes}

            # 4. 변경분 있으면 MongoDB에 폴링 상태 업데이트
            current_time = datetime.now()
            if has_changes:
                mongodb_manager.update_polling_status(current_time)
                logger.info(f"데이터 변경분 감지: {total_changes}개")
            else:
                logger.debug("데이터 변경분 없음")
            # 반환: {"has_changes": bool, "total_changes": int, "changes_by_table": {...}}
            return {
                "has_changes": has_changes,
                "total_changes": total_changes,
                "changes_by_table": changes_by_table,
                "last_check_time": last_check_time.isoformat(),
                "current_time": current_time.isoformat()
            }
        except Exception as e:
            logger.error(f"데이터 변경분 확인 실패: {e}")
            return {
                "has_changes": False,
                "error": str(e),
                "check_time": datetime.now().isoformat()
            }
    def extract_feature_data(self) -> dict: # 호출 관계: ml_pipeline.handle_data_changes() -> 에서 이 메서드 호출
        """feature engineering용 전체 데이터 추출"""
        try:
            # 1. 회원 데이터: member + public_profile + user_preferences JOIN
            member_query = """
            SELECT
                m.id as member_id,
                m.gender as member_gender,
                m.birth_date,
                m.created_at as member_created_at,
                m.updated_at as member_updated_at,
                pp.lifestyle as profile_lifestyle,
                pp.personality as profile_personality,
                pp.is_smoking as profile_is_smoking,
                pp.is_snoring as profile_is_snoring,
                pp.has_pet as profile_has_pet,
                pp.info as profile_info,
                up.cohabitant_count,
                up.has_pet as pref_has_pet,
                up.is_smoking as pref_is_smoking,
                up.is_snoring as pref_is_snoring,
                up.gender as pref_gender,
                up.lifestyle as pref_lifestyle,
                up.personality as pref_personality,
                up.info as pref_info
            FROM member m
            LEFT JOIN public_profile pp ON m.public_profile_id = pp.public_profile_id
            LEFT JOIN user_preferences up ON m.id = up.member_id
            WHERE m.is_completed = true
            """
            # execute_query 사용으로 변경
            members_data = self.execute_query(member_query)
            # 2. 구인글 데이터
            posts_query = """
            SELECT 
                id as post_id,
                user_id,
                title,
                address,
                has_room,
                min_age,
                max_age,
                monthly_cost_min,
                monthly_cost_max,
                rent_cost_min,
                rent_cost_max,
                region_latitude,
                region_longitude,
                recruit_count,
                prefered_gender,
                personality,
                life_style,
                is_smoking,
                is_snoring,
                is_pets_allowed,
                status,
                detail_description,
                additional_description,
                created_at as post_created_at,
                updated_at as post_updated_at
            FROM recruit_post
            """

            # execute_query 사용으로 변경
            posts_data = self.execute_query(posts_query)

            # 3. 상호작용 데이터 (bookmark)
            bookmarks_query = """
            SELECT 
                member_id,
                recruit_post_id,
                created_at as bookmark_created_at,
                updated_at as bookmark_updated_at
            FROM bookmark
            """

            # execute_query 사용으로 변경
            bookmarks_data = self.execute_query(bookmarks_query)

            # 4. 결과 정리
            result = {
                "members": members_data,
                "recruit_posts": posts_data,
                "bookmarks": bookmarks_data,
                "extraction_time": datetime.now().isoformat(),
                "data_counts": {
                    "members": len(members_data),
                    "recruit_posts": len(posts_data),
                    "bookmarks": len(bookmarks_data)
                }
            }
            logger.info(f"Feature 데이터 추출 완료 - 회원: {len(members_data)}, 구인글: {len(posts_data)}, 북마크: {len(bookmarks_data)}")
            return result
        except Exception as e:
            logger.error(f"Feature 데이터 추출 실패: {e}")
            return {
                "members": [],
                "recruit_posts": [],
                "bookmarks": [],
                "error": str(e),
                "extraction_time": datetime.now().isoformat()
            }
    def execute_query(self, query: str, params: tuple = None) -> List[dict]:
        """SQL 쿼리 실행하고 결과 반환 (CloudSQL Proxy 지원)"""
        conn = None
        try:
            conn = self._get_connection()

            if hasattr(self, 'use_proxy') and self.use_proxy:
                # SQLAlchemy 연결 사용 (CloudSQL Proxy)
                if params:
                    # SQLAlchemy는 딕셔너리 형태의 파라미터를 선호
                    if isinstance(params, (tuple, list)):
                        # %s 스타일 파라미터를 :param 스타일로 변환
                        param_names = [f"param{i}" for i in range(len(params))]
                        param_dict = {name: value for name, value in zip(param_names, params)}
                        # 쿼리의 %s를 :param0, :param1 등으로 변경
                        converted_query = query
                        for i, param_name in enumerate(param_names):
                            converted_query = converted_query.replace('%s', f':{param_name}', 1)
                        result = conn.execute(text(converted_query), param_dict)
                    else:
                        result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                # SELECT 쿼리인 경우 결과 반환
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row._mapping) for row in result]
                else:
                    # INSERT, UPDATE, DELETE 등의 경우
                    conn.commit()
                    return []
            else:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                # SELECT 쿼리인 경우 결과 반환
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:
                    # INSERT, UPDATE, DELETE 등의 경우
                    conn.commit()
                    return []
        except Exception as e:
            if conn and not (hasattr(self, 'use_proxy') and self.use_proxy):
                conn.rollback()
            logger.error(f"쿼리 실행 실패: {e}")
            raise
        finally:
            if conn:
                if not (hasattr(self, 'use_proxy') and self.use_proxy):
                    cursor.close()
                self._return_connection(conn)

_postgres_client_instance = None
def get_postgresql_client() -> PostgreSQLClient:
    """PostgreSQL 클라이언트 싱글톤 반환"""
    # config에서 설정값 가져와서 PostgreSQLClient 인스턴스 반환
    global _postgres_client_instance
    if _postgres_client_instance is None:
        config = get_postgres_config()
        if config.get("use_proxy", False):
            # CloudSQL Proxy 사용 시 더미 값 전달 (실제로는 사용되지 않음)
            _postgres_client_instance = PostgreSQLClient(
                host="localhost",
                port=5432, 
                user=config['user'],
                password=config['password'],
                database=config['database']
            )
        else:
            # 기존 방식
            _postgres_client_instance = PostgreSQLClient(
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password'],
                database=config['database']
            )
    return _postgres_client_instance