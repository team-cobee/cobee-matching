"""
    ml pipeline
"""
import logging
from datetime import datetime
from typing import Dict, Optional
from ..database.postgres_client import get_postgresql_client
from ..ml.feature_engineer import get_feature_engineer
from .model_trainer import get_model_trainer
from .recommendation_generator import get_recommendation_generator
from ..database.mongodb_manager import get_mongodb_manager
logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self):
        self.pipeline_status = "initialized"
        self.model_version = "v1.0"
        self.last_training_time = None
        self.current_model = None
        
        # MongoDB에서 현재 상태 로드
        self._sync_with_mongodb()
        logger.info("ML 파이프라인 초기화 완료")
        
    def _sync_with_mongodb(self):
        """MongoDB와 상태 동기화"""
        try:
            mongodb_manager = get_mongodb_manager()
            self.model_version = mongodb_manager.get_current_model_version()
            
            # 최근 학습 로그에서 마지막 학습 시간 가져오기
            training_log = mongodb_manager.get_training_log_by_model_version(self.model_version)
            if training_log:
                training_end = training_log.get('training_end')
                if training_end:
                    # training_end가 이미 datetime 객체인지 문자열인지 확인
                    if isinstance(training_end, str):
                        self.last_training_time = datetime.fromisoformat(training_end)
                    else:
                        # 이미 datetime 객체인 경우
                        self.last_training_time = training_end
                self.pipeline_status = "ready"
                logger.info(f"MongoDB에서 상태 동기화 완료 - 모델: {self.model_version}")
            else:
                logger.info("기존 학습 로그가 없음 - 초기 상태 유지")
                
        except Exception as e:
            # MongoDB 조회 실패 시 기본값 유지
            logger.warning(f"MongoDB 상태 동기화 실패, 기본값 사용: {e}")
            self.model_version = "v1.0"

    def get_pipeline_status(self) -> dict:
        """
        파이프라인 상태 조회 (실시간 MongoDB 조회로 정확한 상태 반환)
          - last_training: 가장 최근 훈련 시도가 언제 있었는지 (성공/실패 관계없이)
        """
        try:
            # 실시간으로 MongoDB에서 현재 모델 버전 조회
            mongodb_manager = get_mongodb_manager()
            current_model_version = mongodb_manager.get_current_model_version()
        except Exception as e:
            logger.warning(f"MongoDB 모델 버전 조회 실패, 캐시된 값 사용: {e}")
            current_model_version = self.model_version
            
        return {
            "pipeline_status": self.pipeline_status,  # "initialized", "training", "ready", "error"
            "model_info": {
                "model_version": current_model_version,  # 실시간 조회 값
                "model_type": "lightgbm",
            },
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None
        }
    def train_model(self, force_retrain: bool = False) -> dict:
        """모델 훈련 실행"""
        try:
            self.pipeline_status = "training"
            self.last_training_time = datetime.now()
            
            logger.info(f"수동 모델 학습 시작 (force_retrain={force_retrain})")
            
            # handle_data_changes 호출하여 전체 파이프라인 실행
            pipeline_result = self.handle_data_changes()
            
            if pipeline_result.get("status") == "complete":
                self.pipeline_status = "ready"
                
                # MongoDB에서 최신 학습 로그 조회
                mongodb_manager = get_mongodb_manager()
                current_model_version = mongodb_manager.get_current_model_version()
                training_log = mongodb_manager.get_training_log_by_model_version(current_model_version)
                
                return {
                    "status": "completed",
                    "training_id": training_log.get("training_id") if training_log else None,
                    "model_version": current_model_version,
                    "metrics": training_log.get("metrics") if training_log else {},
                    "started_at": self.last_training_time.isoformat(),
                    "completed_at": datetime.now().isoformat()
                }
            else:
                self.pipeline_status = "error"
                return {
                    "status": "failed",
                    "error": pipeline_result.get("error"),
                    "started_at": self.last_training_time.isoformat(),
                    "failed_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.pipeline_status = "error"
            logger.error(f"모델 학습 실패: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "started_at": self.last_training_time.isoformat() if self.last_training_time else None,
                "failed_at": datetime.now().isoformat()
            }

    # 폴링에서 호출되는 메인 로직
    def handle_data_changes(self) -> dict:
        """데이터 처리 메인 로직(전체 재학습 진행할 예정)"""
        try:
            logger.info("=== ML 파이프라인 시작 ===")
            # 1. PostgreSQL에서 feature engineering용 데이터 추출
            postgres_client = get_postgresql_client()
            raw_data = postgres_client.extract_feature_data()

            logger.info(f"원시 데이터 추출 완료 - 회원: {raw_data['data_counts']['members']}개")
            # 2. Feature engineering 진행

            logger.info("Feature Engineering 시작")
            feature_engineer = get_feature_engineer()

            feature_data = feature_engineer.process_full_data(raw_data)
            
            logger.info(f"Feature Engineering 완료 - {len(feature_data['features'])}개 샘플, {len(feature_data['feature_names'])}개 특성")
            logger.info(f"생성된 특성들: {', '.join(feature_data['feature_names'][:10])}{'...' if len(feature_data['feature_names']) > 10 else ''}")

            # 긍정 샘플(북마크) 비율 로그
            features_df = feature_data['features']
            positive_rate = features_df['is_bookmarked'].mean()
            logger.info(f"긍정 샘플 비율: {positive_rate:.3f} ({features_df['is_bookmarked'].sum()}개/{len(features_df)}개)")
            
            # 3. 모델 학습 및 재훈련
            # TODO : model_trainer.train_full_model()
            logger.info("모델 학습 시작")

            model_trainer = get_model_trainer()
            training_result = model_trainer.train_full_model(feature_data)

            logger.info(f"모델 학습 완료 - 모델 버전: {training_result.get('model_version', 'unknown')}")
            logger.info(f"학습 성능: AUC={training_result['metrics'].get('auc',0):.3f}")

            if training_result.get('status') == 'replaced': # 5. 추천 결과 생성 및 MongoDB 저장 (모델이 교체된 경우만)
                logger.info(f"새 모델로 교체됨: {training_result.get('model_version')}")
                rec_generator = get_recommendation_generator()
                recommendations = rec_generator.generate_all_recommendations(
                    training_result['model'],
                    feature_data
                )

                mongodb_manager = get_mongodb_manager()
                mongodb_manager.save_batch_recommendations(recommendations)
                logger.info(f"추천 결과 저장 완료: {len(recommendations)}명")

            elif training_result.get('status') == 'rejected':
                logger.info(f"성능 개선 없어 모델 교체 안함: {training_result.get('reason')}")                 
            
            logger.info("=== ML 파이프라인 완료 ===")
            return {
                "status": "complete",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"ML 파이프라인 실패 : {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

_ml_pipeline_instance = None
def get_ml_pipeline() -> MLPipeline:
    """ML 파이프라인 싱글톤 반환"""
    global _ml_pipeline_instance
    if _ml_pipeline_instance is None:
        _ml_pipeline_instance = MLPipeline()
    return _ml_pipeline_instance