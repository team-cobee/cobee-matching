"""
    추천 생성
"""
import logging
from typing import List
import pandas as pd
from datetime import datetime
from ..database.mongodb_manager import get_mongodb_manager
from ..database.postgres_client import get_postgresql_client
logger = logging.getLogger(__name__)

class RecommendationGenerator:
    def __init__(self):
        pass

    def generate_all_recommendations(self, model, feature_data: dict) -> List[dict]:
        """모든 회원에 대한 추천 결과 생성"""
        try:
            
            logger.info("모든 회원 추천 결과 생성 시작...")
            
            # 1. feature_data에서 필요한 정보 추출
            features_df = feature_data['features']
            feature_names = feature_data['feature_names']
            
            # 2. 모든 회원-구인글 조합에 대해 예측 수행
            X = features_df[feature_names]
            # 북마크할 확률 예측 (class=1의 확률)
            prediction_scores = model.predict_proba(X)[:, 1]
            
            # 3. 예측 결과를 원본 데이터프레임에 추가
            features_df = features_df.copy()
            features_df['prediction_score'] = prediction_scores
            
            # 4. 회원별로 추천 결과 생성
            recommendations = []
            unique_members = features_df['member_id'].unique()
            
            mongodb_manager = get_mongodb_manager()
            current_model_version = mongodb_manager.get_current_model_version()
            
            for member_id in unique_members:
                # 해당 회원의 모든 구인글-점수 조합 조회
                member_data = features_df[features_df['member_id'] == member_id]
                
                # 자신의 구인글 제외 (is_own_post == 0인 것만)
                member_data = member_data[member_data['is_own_post'] == 0]
                
                # 예측 점수 기준으로 내림차순 정렬 (모든 구인글)
                top_recommendations = member_data.sort_values('prediction_score', ascending=False)
                
                # 추천할 구인글이 있는 경우만 저장
                if len(top_recommendations) > 0:
                    # MongoDB 스키마에 맞는 형태로 변환
                    recommendation_list = []
                    for _, row in top_recommendations.iterrows():
                        post_details = self._get_post_details(row['recruit_post_id']) # _get_post_details

                        recommendation_list.append({
                            "recruit_post_id": int(row['recruit_post_id']),
                            "score": float(row['prediction_score']),
                            "post_details": {
                                "title": post_details.get('titile'),
                                "address": post_details.get('address'),
                                "has_room": post_details.get('has_room'),
                                "monthly_cost_min": post_details.get('monthly_cost_min'),
                                "monthly_cost_max": post_details.get('monthly_cost_max')
                            },
                            "created_at": datetime.now().isoformat()
                        })
                    
                    recommendations.append({
                        "recommendation_id": f"rec_{member_id}_{current_model_version}",
                        "member_id": int(member_id),
                        "recommendations": recommendation_list,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "model_version": current_model_version
                    })
            
            logger.info(f"추천 결과 생성 완료 - {len(recommendations)}명, 평균 {sum(len(r['recommendations']) for r in recommendations) / len(recommendations) if recommendations else 0:.1f}개 추천")
            return recommendations
            
        except Exception as e:
            logger.error(f"추천 결과 생성 실패: {e}")
            return []

    def _get_all_members(self) -> List[int]:
        """모든 회원 목록 조회 (모두 활성 취급)"""
        # 내부 메서드: PostgreSQL에서 전체 회원 조회
        pass

    def _get_active_posts(self) -> List[int]:
        """현재 활성 구인글 목록 조회"""
        # 내부 메서드: PostgreSQL에서 활성 구인글 조회
        pass

    def _predict_for_member(self, model, member_id: int, post_ids: List[int]) -> List[dict]:
        """특정 회원에 대한 추천 예측"""
        # 내부 메서드: 모델을 사용해 회원-구인글 점수 예측
        pass

    def _get_post_details(self, post_id: int) -> dict:
        """구인글 상세 정보 조회"""
        try:
            postgres_client = get_postgresql_client()

            post_id = int(post_id)
            # PostgreSQL에서 구인글 상세 정보 조회
            query = """
            SELECT title, address, has_room, monthly_cost_min, monthly_cost_max
            FROM recruit_post
            WHERE id = %s
            """

            result = postgres_client.execute_query(query, (post_id,))
            if result and len(result) > 0:
                post_data = result[0]
                return {
                    "title": post_data.get('titile', ''),
                    "address": post_data.get('address', ''),
                    "has_room": post_data.get('has_room', False),
                    "monthly_cost_min": post_data.get('monthly_cost_min', 0),
                    "monthly_cost_max": post_data.get('monthly_cost_max', 0)                    
                }
            else:
                logger.warning(f"구인글 {post_id}의 상세 정보를 찾을 수 없습니다.")
                return {}
        except Exception as e:
            logger.error(f"구인글 {post_id} 상세 정보 조회 실패: {e}")
            return {}
_recommendation_generator_instance = None

def get_recommendation_generator() -> RecommendationGenerator:
    """RecommendationGenerator 싱글톤 반환"""
    global _recommendation_generator_instance
    if _recommendation_generator_instance is None:
        _recommendation_generator_instance = RecommendationGenerator()
    return _recommendation_generator_instance