"""
    모델 훈련 모듈 - LightGBM을 사용한 추천 시스템 모델 학습
"""
import pandas as pd
import numpy as np
# lightgbm: Microsoft의 고성능 gradient boosting 라이브러리
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import pickle
import logging
from datetime import datetime
from ..database.mongodb_manager import get_mongodb_manager

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model_save_dir = "models"
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train_full_model(self, feature_data: dict) -> dict:
        """전체 데이터로 모델 훈련을 실행하는 메인 메서드"""
        # 출력: 학습된 모델과 성능 지표를 포함한 딕셔너리
        try:
            logger.info("LightGBM 모델 학습 시작...")

            # 1. 입력 데이터에서 필요한 정보 추출
            features_df = feature_data['features']
            feature_names = feature_data['feature_names']

            # 2. 머신러닝을 위한 X(입력)와 y(정답) 분리
            X = features_df[feature_names]
            y = features_df['is_bookmarked']

            # 3. 데이터를 훈련용(80%)과 검증용(20%)으로 분할
            # stratify=y: 긍정/부정 비율을 훈련/검증에서 동일하게 유지
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info(f"데이터 분할 완료 - 훈련: {len(X_train)}개, 검증: {len(X_val)}개")
            
            # 4. 실제 LightGBM 모델 학습 수행
            model = self._train_lightgbm(X_train, y_train, X_val, y_val)
            
            # 5. 학습된 모델의 성능을 검증 데이터로 평가
            metrics = self._evaluate_model(model, X_val, y_val)

            # 6. 기존 모델과 성능 비교
            should_replace = self._compare_with_current_model(metrics)

            if should_replace:
                # 7a. 성능이 더 좋으면 모델 교체
                # 모델 버전: 현재 시간을 기반으로 고유 버전 생성 (예: v20241220_143022)
                model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # 모델을 파일로 저장하고 저장 경로 반환
                model_path = self._save_model(model, model_version)
                self._replace_current_model(model_version)
                # 학습 결과를 MongoDB에 저장
                training_info = {
                    "total_members": len(features_df['member_id'].unique()),
                    "total_posts": len(features_df['recruit_post_id'].unique()),
                    "total_interactions": len(features_df),
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val)
                }
                mongodb_manager = get_mongodb_manager()
                training_id = mongodb_manager.save_training_log(model_version, metrics, training_info)
                # 8. 학습 결과를 딕셔너리로 반환
                return {
                    "status": "replaced",
                    "model": model,                    # 학습된 LightGBM 모델 객체
                    "model_version": model_version,    # 모델 버전 (시간 기반)
                    "model_path": model_path,          # 저장된 모델 파일 경로
                    "metrics": metrics,                # 성능 지표들 (AUC, Precision 등)
                    "feature_names": feature_names,    # 사용된 특성 이름들
                    "training_samples": len(X_train),  # 훈련에 사용된 샘플 수
                    "validation_samples": len(X_val),   # 검증에 사용된 샘플 수              
                    "training_id": training_id  # MongoDB 저장 ID 추가
                }
            else:
                # 7b. 성능이 나쁘면 버림
                return {"status": "rejected", "reason": "performance_lower","metrics": metrics}
        except Exception as e:
            logger.error(f"모델 학습 실패 : {e}")
            raise e
        
    def _compare_with_current_model(self, new_metrics: dict) -> bool:
        """새 모델과 현재 모델 성능 비교"""
        # MongoDB에서 현재 모델 버전 조회
        # 현재 모델 성능 로그 조회
        # 성능 비교 로직 (AUC 우선순위)
        try:
            mongodb_manager = get_mongodb_manager()

            # 1. 현재 사용중인 모델 버전 조회
            current_model_version = mongodb_manager.get_current_model_version()
            logger.info(f"현재 모델 버전: {current_model_version}")

            # 2, 현재 모델의 성능 로그 조회
            current_log = mongodb_manager.get_training_log_by_model_version(current_model_version)
            
            # 3. 현재 모델이 없는 경우 (최초 실행)
            if not current_log:
                logger.info("현재 모델이 없음. 새 모델을 현재 모델로 설정")
                return True
            # 4. 성능 비교 로직 (우선순위: AUC > Precision@10 > PR-AUC)
            current_metrics = current_log.get('metrics', {})

            # AUC 비교 (1순위)
            new_auc = new_metrics.get('auc', 0)
            current_auc = current_metrics.get('auc', 0)

            if new_auc > current_auc:
                logger.info(f"AUC 개선: {current_auc:.3f} → {new_auc:.3f}")
                return True
            elif new_auc < current_auc:
                logger.info(f"AUC 하락: {current_auc:.3f} → {new_auc:.3f}")
                return False

            # AUC가 동일한 경우 Precision@10 비교 (2순위)
            new_p10 = new_metrics.get('precision_at_10', 0)
            current_p10 = current_metrics.get('precision_at_10', 0)

            if new_p10 > current_p10:
                logger.info(f"Precision@10 개선: {current_p10:.3f} → {new_p10:.3f}")
                return True
            elif new_p10 < current_p10:
                logger.info(f"Precision@10 하락: {current_p10:.3f} → {new_p10:.3f}")
                return False

            # 둘 다 동일한 경우 PR-AUC 비교 (3순위)
            new_pr_auc = new_metrics.get('pr_auc', 0)
            current_pr_auc = current_metrics.get('pr_auc', 0)

            if new_pr_auc > current_pr_auc:
                logger.info(f"PR-AUC 개선: {current_pr_auc:.3f} → {new_pr_auc:.3f}")
                return True
            else:
                logger.info("모든 지표에서 개선되지 않음")
                return False

        except Exception as e:
            logger.error(f"모델 성능 비교 실패: {e}")
            # 에러 발생 시 안전하게 False 반환 (기존 모델 유지)
            return False            
    
    def _replace_current_model(self, new_model_version: str):
        """현재 모델을 새 모델로 교체"""       
        try:
            mongodb_manager = get_mongodb_manager()
            # polling_status의 current_model_version 업데이트 
            mongodb_manager.db.polling_status.update_one(
                {"_id": "polling_status"},
                {
                    "$set": {
                      "current_model_version": new_model_version,
                      "last_check": datetime.now(),  # README 스키마 준수
                      "status": "running"           # README 스키마 준수                  
                    }
                },
                upsert=True
            )
            logger.info(f"현재 모델이 {new_model_version}으로 교체되었습니다")
        except Exception as e:
            logger.error(f"모델 교체 실패: {e}")
            raise


    def _train_lightgbm(self, X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
        """LightGBM 알고리즘을 사용하여 실제 모델을 학습하는 메서드"""
        
        # LightGBM 하이퍼파라미터 설정
        params = {
            # 이진 분류 문제로 설정 (북마크 할/안할의 2개 클래스)
            'objective': 'binary',
            # 모델 성능 평가 지표로 AUC 사용 (0.5~1.0, 높을수록 좋음)
            'metric': 'auc',
            # Gradient Boosting Decision Tree 방식 사용
            'boosting_type': 'gbdt',
            # 각 트리의 최대 잎 노드 개수 (31은 적당한 복잡도)
            'num_leaves': 31,
            # 학습률 - 낮을수록 안정적이지만 느림 (0.05는 보수적 설정)
            'learning_rate': 0.05,
            # 각 트리에서 사용할 특성의 비율 (90% 사용, 오버피팅 방지)
            'feature_fraction': 0.9,
            # 각 트리에서 사용할 데이터의 비율 (80% 사용, 오버피팅 방지)
            'bagging_fraction': 0.8,
            # bagging을 수행할 주기 (5번마다 데이터 재샘플링)
            'bagging_freq': 5,
            # 학습 과정에서 출력할 로그 레벨 (-1: 출력 안함)
            'verbose': -1,
            'random_state': 42,
            # 최대 트리 개수 (1000개까지 학습, early stopping으로 조기 종료 가능)
            'n_estimators': 1000
        }
        
        # 불균형 데이터 처리: 북마크 안한 샘플 수 / 북마크 한 샘플 수
        # 예: 9000개 vs 1000개 → 가중치 9.0 (북마크 데이터를 9배 더 중요하게 학습)
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        # 계산된 가중치를 파라미터에 추가
        params['scale_pos_weight'] = pos_weight

        logger.info(f"긍정 샘플 가중치: {pos_weight:.2f}")

        # LightGBM 분류기 객체 생성
        # 내부 로직: 약한 학습기(decision tree)들을 순차적으로 학습하여
        # 이전 모델의 오류를 다음 모델이 보완하는 방식으로 성능 향상
        model = lgb.LGBMClassifier(**params)

        # 실제 모델 학습 수행
        model.fit(
            X_train, y_train,                    # 훈련 데이터 (특성, 라벨)
            eval_set=[(X_val, y_val)],          # 검증 데이터 (학습 중 성능 모니터링용)
            eval_metric='auc',                  # 검증 성능 평가 지표
            callbacks=[
                lgb.early_stopping(50),        # 50회 동안 성능 개선 없으면 학습 중단
                lgb.log_evaluation(100)        # 100회마다 학습 진행 상황 출력
            ]
        )
        logger.info("LightGBM 모델 학습 완료")
        return model
    
    def _evaluate_model(self, model, X_val, y_val) -> dict:
        """학습된 모델의 성능을 다양한 지표로 평가하는 메서드"""
        # TODO: 모델 성능을 MongoDB에 저장하는 로직 추가 필요
        
        # 모델 예측 수행
        # predict_proba(): 각 클래스에 속할 확률을 반환 [[P(class=0), P(class=1)], ...]
        # [:, 1]: 북마크할 확률(class=1)만 선택하여 추출
        # 결과: 각 회원-구인글 조합이 북마크될 확률 (0.0~1.0)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # ROC-AUC 점수 계산
        # AUC: Area Under Curve (ROC 곡선 아래 면적)
        # 0.5~1.0 범위, 0.5는 랜덤, 1.0은 완벽한 분류
        # 임계값에 관계없이 모델의 전반적인 분류 성능을 측정
        auc_score = roc_auc_score(y_val, y_pred_proba)

        # Precision-Recall AUC 계산
        # precision_recall_curve(): 다양한 임계값에서 precision과 recall 계산
        # precision: 모델이 긍정으로 예측한 것 중 실제 긍정인 비율
        # recall: 실제 긍정 중 모델이 올바르게 예측한 비율
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        # PR-AUC: Precision-Recall 곡선 아래 면적 (불균형 데이터에서 더 의미있는 지표)
        pr_auc = auc(recall, precision)

        # Precision@K 계산 (추천 시스템의 핵심 성능 지표)
        # 상위 K개 추천 중 실제로 관심있어할 것의 비율
        # 예: 10개 추천 중 3개를 북마크 → P@10 = 30%
        # 추천 시스템에서는 상위 몇 개만 보므로 가장 중요한 지표
        top_k_precision = self._calculate_precision_at_k(y_val, y_pred_proba, k=10)
        
        # 성능 지표들을 딕셔너리로 정리 (numpy 타입을 Python 기본 타입으로 변환)
        metrics = {
            'auc': float(auc_score),                    # ROC-AUC 점수
            'pr_auc': float(pr_auc),                   # Precision-Recall AUC
            'precision_at_10': float(top_k_precision), # 상위 10개 추천 정확도
            'positive_samples': int(y_val.sum()),      # 검증 데이터 중 긍정 샘플 수
            'total_samples': int(len(y_val))           # 총 검증 샘플 수
        }
        # 성능 지표들을 로그에 기록
        logger.info(f"모델 성능 - AUC: {auc_score:.3f}, PR-AUC: {pr_auc:.3f}, P@10: {top_k_precision:.3f}")
        return metrics

    def _calculate_precision_at_k(self, y_true, y_scores, k=10):
        """상위 K개 추천의 정확도를 계산하는 메서드 (추천 시스템 핵심 지표)"""
        # numpy.argsort(): 배열을 정렬했을 때의 인덱스를 반환
        # [::-1]: 내림차순으로 정렬 (높은 점수부터)
        # [:k]: 상위 K개만 선택
        # 결과: 예측 확률이 높은 순서대로 K개의 인덱스
        sorted_indices = np.argsort(y_scores)[::-1][:k]
        
        # y_true가 pandas Series인지 numpy array인지 확인하여 적절히 인덱싱
        # iloc: pandas Series의 위치 기반 인덱싱
        # 결과: 상위 K개 중 실제 북마크된 것들의 라벨 (0 또는 1)
        top_k_true = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
        
        # 평균을 계산하여 Precision@K 반환
        # 예: [1, 0, 1, 0, 1, 0, 0, 0, 0, 0] → 3/10 = 0.3 (30%)
        return top_k_true.mean()
    
    def _save_model(self, model, model_version: str) -> str:
        """학습된 모델을 파일 시스템에 저장하는 메서드"""
        # 모델 파일명 생성: "lightgbm_model_v20241220_143022.pkl" 형태
        model_filename = f"lightgbm_model_{model_version}.pkl"
        # 모델 저장 디렉토리와 파일명을 결합하여 전체 경로 생성
        model_path = os.path.join(self.model_save_dir, model_filename)

        # pickle을 사용하여 모델 객체를 바이너리 파일로 저장
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"모델 저장 완료 : {model_path}")
        return model_path
    


_model_trainer_instance = None

def get_model_trainer() -> ModelTrainer:
    """ModelTrainer 객체를 싱글톤으로 반환하는 팩토리 함수"""
    global _model_trainer_instance
    if _model_trainer_instance is None:
        _model_trainer_instance = ModelTrainer()
    return _model_trainer_instance