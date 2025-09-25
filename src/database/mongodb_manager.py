"""
  MongoDB 매니저 - 추천 시스템 데이터 관리
"""
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import logging
from datetime import datetime
from typing import Dict, List, Optional
from .config import get_mongodb_config

logger = logging.getLogger(__name__)
class MongoDBManager:
    def __init__(self, mongodb_uri: str, database_name: str):
        """MongoDB 매니저 초기화"""
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.client = None
        self.db = None
        self._connect()
        self.create_collections()
        logger.info("MongoDB 매니저 초기화 완료")
    
    def _connect(self) -> None:
        """MongoDB 연결 설정"""
        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            # 연결 테스트
            self.client.admin.command('ping')
            logger.info(f"MongoDB 연결 성공 : {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            raise

    def create_collections(self) -> None:
        """필요한 컬렉션 및 인덱스 생성"""
        # 컬렉션이 없을 때만 생성, 기존 있으면 무시, 인덱스만 확인/추가
        try:
            # 1. recommendations 컬렉션
            if 'recommendations' not in self.db.list_collection_names():
                self.db.create_collection('recommendations')
                # 인덱스 : 검색 성능을 위한 것
                self.db.recommendations.create_index('member_id', unique=True)
                self.db.recommendations.create_index("updated_at")
            
            # 2. training_logs 컬렉션
            if 'training_logs' not in self.db.list_collection_names():
                self.db.create_collection('training_logs')
                
                # training_logs 인덱스
                self.db.training_logs.create_index("training_start")
                self.db.training_logs.create_index("model_version")
            # 3. recommendation_history 컬렉션
            if 'recommendation_history' not in self.db.list_collection_names():
                self.db.create_collection('recommendation_history')

                self.db.recommendation_history.create_index([('member_id', pymongo.ASCENDING), ('recommended_at', pymongo.DESCENDING)])
            # 4. polling_status 컬렉션
            if 'polling_status' not in self.db.list_collection_names():
                self.db.create_collection('polling_status')
                
                # 초기 polling_status 문서 생성
                self._initialize_polling_status()
            
            # 5. realtime_cache 컬렉션 (실시간 추천 캐싱용)
            if 'realtime_cache' not in self.db.list_collection_names():
                self.db.create_collection('realtime_cache')
                
                # realtime_cache 인덱스
                self.db.realtime_cache.create_index("member_id", unique=True)
                self.db.realtime_cache.create_index("expires_at", expireAfterSeconds=0)  # TTL 인덱스
                
            logger.info("MongoDB 컬렉션 및 인덱스 생성 완료")
        except Exception as e:
            logger.error(f"MongoDB 컬렉션 및 인덱스 생성 실패: {e}")
            raise

    def _initialize_polling_status(self) -> None:
        """polling_status 컬렉션 초기 문서 생성"""
        try:
            initial_status = {
                "_id": "polling_status",
                "last_check": datetime.now(),
                "last_data_change": datetime(2020, 1, 1),  # 데이터 변경분 체크용
                "status": "initialized",
                "current_model_version": "v1.0"
            }
            self.db.polling_status.insert_one(initial_status)
            logger.info("polling_status 초기 문서 생성 완료")
        except DuplicateKeyError:
            logger.info("polling_status 초기 문서 이미 존재")
        except Exception as e:
            logger.error(f"polling_status 초기 문서 생성 실패: {e}")
            raise

    def health_check(self) -> dict:
        """MongoDB 헬스체크"""
        try:
            # 연결 테스트
            self.client.admin.command('ping')

            # collection 개수 확인
            collections = self.db.list_collection_names()

            return {
                "mongodb_connected": True,
                "database_info": {
                    "database_name": self.database_name,
                    "collections": collections,
                    "collection_count": len(collections)
                },
                "checked_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"MongoDB 헬스체크 실패: {e}")
            return {
                "mongodb_connected": False,
                "error": str(e),
                "checked_at": datetime.now().isoformat()
            }
    
    def disconnect(self) -> None:
        """MongoDB 연결 종료"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.db = None
                logger.info("MongoDB 연결 종료 완료")
        except Exception as e:
            logger.error(f"MongoDB 연결 종료 실패: {e}")
    
    def get_current_model_version(self) -> str:
        status = self.db.polling_status.find_one({"_id": "polling_status"})
        return status.get("current_model_version", "v1.0") if status else "v1.0"
    
    def get_training_log_by_model_version(self, model_version: str) -> Optional[dict]:
        """특정 모델 버전의 훈련 로그 조회"""
        try:
            log = self.db.training_logs.find_one({"model_version": model_version})
            if log:
                log['_id'] = str(log['_id'])
                logger.info(f"모델 버전 {model_version} 훈련 로그 조회 완료")
                return log
            else:
                logger.info(f"모델 버전 {model_version}의 훈련 로그를 찾을 수 없습니다")
                return None
        except Exception as e:
            logger.error(f"훈련 로그 조회 실패: {e}")
            return None 
                   
    def get_member_recommendations(self, member_id: int) -> List[dict]:
        """회원에게 recruit_post 추천 목록 조회(점수 포함)"""
        try:
            # recommendations 컬렉션에서 해당 회원의 추천 데이터 조회
            recommendation_doc = self.db.recommendations.find_one({"member_id": member_id})

            if not recommendation_doc:
                logger.info(f"회원 {member_id}의 추천 데이터가 없습니다")
                return []
            # recommendations 배열에서 member의 모든 추천 데이터 반환
            all_recommendations = recommendation_doc.get('recommendations', [])
            logger.info(f"회원 {member_id} 전체 추천 조회 완료 : {len(all_recommendations)}개")
            return all_recommendations
        except Exception as e:
            logger.error(f"회원 추천 조회 실패: {e}")
            return []

    def get_recent_training_logs(self, limit: int) -> List[dict]:
        """최근 훈련 로그들 조회"""
        try:
            # training_start 기준으로 내림차순 정렬하여 최근 로그부터 조회
            logs = list(self.db.training_logs.find().sort("training_start", -1).limit(limit))

            # ObjectId를 문자열로 반환
            for log in logs:
                log['_id'] = str(log['_id'])
            logger.info(f"최근 훈련 로그 {len(logs)}개 조회 완료")
            return logs
        except Exception as e:
            logger.error(f"훈련 로그 조회 실패: {e}")
            return []
    def save_recommendation_analysis(self, member_id: int, analysis_data: dict) -> bool:
        """추천 분석 결과 저장 (임계값, 점수 분포 등)"""
        try:
            analysis_record = {
                "member_id": member_id,
                "analysis_data": analysis_data,
                "created_at": datetime.now().isoformat()
            }
            
            # recommendation_analysis 컬렉션에 저장
            self.db.recommendation_analysis.insert_one(analysis_record)
            logger.info(f"회원 {member_id} 추천 분석 결과 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"추천 분석 결과 저장 실패: {e}")
            return False
    def get_member_recommendation_history(self, member_id: int, limit: int) -> List[dict]:
        """회원의 추천 기록 조회"""
        try:
            # recommended_at 기준으로 내림차순 정렬하여 최근 기록부터 조회
            history = list(
                self.db.recommendation_history.find({"member_id": member_id})
                .sort("recommended_at", -1)
                .limit(limit)
            )
            
            # ObjectId를 문자열로 변환
            for record in history:
                record['_id'] = str(record['_id'])
                
            logger.info(f"회원 {member_id} 추천 기록 {len(history)}개 조회 완료")
            return history
            
        except Exception as e:
            logger.error(f"회원 {member_id} 추천 기록 조회 실패: {e}")
            return []

    def save_batch_recommendations(self, recommendations: List[dict]) -> bool:
        """배치로 생성된 추천 결과들을 MongoDB에 저장"""
        try:
            if not recommendations:
                logger.warning("저장할 추천 결과가 없습니다")
                return True
            
            # 기존 추천 결과 삭제 (전체 교체 방식)
            delete_result = self.db.recommendations.delete_many({})
            logger.info(f"기존 추천 결과 {delete_result.deleted_count}개 삭제")
            
            # 새로운 추천 결과 일괄 삽입
            insert_result = self.db.recommendations.insert_many(recommendations)
            
            logger.info(f"새로운 추천 결과 {len(insert_result.inserted_ids)}개 저장 완료")
            
            # recommendation_history에도 개별 기록 저장
            history_records = []
            for recommendation in recommendations:
                member_id = recommendation['member_id']
                model_version = recommendation['model_version']
                
                for rec in recommendation['recommendations']:
                    history_records.append({
                        "member_id": member_id,
                        "recommended_post_id": rec['recruit_post_id'],
                        "score": rec['score'],
                        "recommended_at": rec['created_at'],
                        "model_version": model_version
                    })
            
            if history_records:
                self.db.recommendation_history.insert_many(history_records)
                logger.info(f"추천 히스토리 {len(history_records)}개 기록 저장")
            
            return True
            
        except Exception as e:
            logger.error(f"추천 결과 저장 실패: {e}")
            return False
    def save_training_log(self, model_version: str, metrics: dict, training_info: dict) -> str:
        """훈련 로그 저장"""
        try:
            # README의 training_logs 스키마에 맞춰 문서 생성
            training_log = {
                "training_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model_version": model_version,
                "training_start": datetime.now(),
                "training_end": datetime.now(),  # 실제로는 학습 시작/종료 시간 추적 필요
                "metrics": {
                    "auc": metrics.get('auc', 0),
                    "pr_auc": metrics.get('pr_auc', 0),
                    "precision_at_10": metrics.get('precision_at_10', 0),
                    "positive_samples": metrics.get('positive_samples', 0),
                    "total_samples": metrics.get('total_samples', 0)
                },
                "data_info": {
                    "total_members": training_info.get('total_members', 0),
                    "total_posts": training_info.get('total_posts', 0),
                    "total_interactions": training_info.get('total_interactions', 0),
                    "training_samples": training_info.get('training_samples', 0),
                    "validation_samples": training_info.get('validation_samples', 0)
                },
                "status": "completed"
            }
            
            # MongoDB에 저장
            result = self.db.training_logs.insert_one(training_log)
            logger.info(f"훈련 로그 MongoDB 저장 완료: {training_log['training_id']}")
            
            return training_log['training_id']
        except Exception as e:
            logger.error(f"훈련 로그 저장 실패: {e}")
            raise
    
    def get_last_check_time(self) -> datetime: # polling_status 레코드는 오직 한 개만 존재
        """마지막 체크 시간 조회"""
        try:
            status = self.db.polling_status.find_one({"_id": "polling_status"})
            if status:
                last_change = status.get("last_data_change")
                if last_change:
                    return last_change
                else:
                    # 최초 실행: 과거 시간으로 설정하여 모든 데이터 감지
                    return datetime(2020, 1, 1)  # 충분히 과거
            else:
                # 문서가 없으면 최초 실행
                return datetime(2020, 1, 1)
        except Exception as e:
            logger.error(f"마지막 체크 시간 조회 실패: {e}")
            return datetime(2020, 1, 1)  # 실패해도 과거 시간
        
    def update_polling_status(self, check_time: datetime) -> None: # 변경분 감지 메서드에서 check_time 전달
        """폴링 상태 업데이트"""
        try:
            self.db.polling_status.update_one(
                {"_id": "polling_status"},
                {
                    "$set": {
                        "last_check": datetime.now(),
                        "last_data_change": check_time,
                        "status": "running",
                        "current_model_version": "v1.0"  # TODO: 실제 모델 버전으로 교체
                    }
                },
                upsert=True
            )
            logger.debug(f"폴링 상태 업데이트 완료: {check_time}")
        except Exception as e:
            logger.error(f"폴링 상태 업데이트 실패: {e}")

_mongodb_manager_instance = None
def get_mongodb_manager() -> MongoDBManager:
    """MongoDB 매니저 싱글톤 반환"""
    global _mongodb_manager_instance
    if _mongodb_manager_instance is None:
        config = get_mongodb_config()
        _mongodb_manager_instance = MongoDBManager(
            mongodb_uri=config['url'],
            database_name=config['database']
        )
    return _mongodb_manager_instance