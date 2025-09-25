"""
실시간 추천 생성기 - 신규 회원용 (MongoDB 캐싱 활용)
"""
import logging
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from ..database.postgres_client import get_postgresql_client
from ..database.mongodb_manager import get_mongodb_manager
from .feature_engineer import get_feature_engineer
import joblib
import os

logger = logging.getLogger(__name__)

class RealtimeRecommender:
    def __init__(self):
        self.model = None
        self.feature_engineer = get_feature_engineer()
        self._load_current_model()
    
    def _load_current_model(self):
        """현재 모델 로드"""
        try:
            # MongoDB에서 현재 모델 버전 확인
            mongodb_manager = get_mongodb_manager()
            current_version = mongodb_manager.get_current_model_version()
            
            # 모델 파일 경로 (models/ 디렉토리에서)
            model_path = f"models/lightgbm_model_{current_version}.pkl"
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"실시간 추천용 모델 로드 완료: {current_version}")
            else:
                # 기본 모델 사용
                model_path = "models/lightgbm/lightgbm_model.txt"
                if os.path.exists(model_path):
                    import lightgbm as lgb
                    self.model = lgb.Booster(model_file=model_path)
                    logger.info("기본 LightGBM 모델 로드 완료")
                else:
                    logger.error("사용 가능한 모델을 찾을 수 없습니다")
                    
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.model = None
    
    def generate_realtime_recommendations(self, member_id: int, limit: int = 10) -> List[Dict]:
        """
        신규 회원을 위한 실시간 추천 생성 (MongoDB 캐싱 활용)
        """
        try:
            # 1. MongoDB에서 캐시된 실시간 추천 확인 (10분 TTL)
            cached_recommendations = self._get_cached_realtime_recommendations(member_id)
            if cached_recommendations:
                logger.info(f"회원 {member_id} 캐시된 실시간 추천 반환: {len(cached_recommendations)}개")
                return cached_recommendations[:limit]
            
            if not self.model:
                logger.error("모델이 로드되지 않아 실시간 추천 불가")
                return []
            
            logger.info(f"회원 {member_id} 실시간 추천 생성 시작")
            
            # 2. 해당 회원의 데이터 조회
            member_data = self._get_member_data(member_id)
            if not member_data:
                logger.warning(f"회원 {member_id} 데이터를 찾을 수 없습니다")
                return []
            
            # 3. 현재 활성 구인글 조회 (자신의 구인글 제외)
            active_posts = self._get_active_posts_for_member(member_id)
            if not active_posts:
                logger.info(f"회원 {member_id}에게 추천할 활성 구인글이 없습니다")
                return []
            
            # 4. 회원-구인글 조합으로 feature engineering
            member_post_pairs = []
            for post in active_posts:
                member_post_pairs.append({
                    'member_id': member_id,
                    'recruit_post_id': post['post_id'],
                    'member_data': member_data,
                    'post_data': post
                })
            
            # 5. Feature 생성
            features_df = self._create_features_for_pairs(member_post_pairs)
            
            if features_df.empty:
                logger.warning(f"회원 {member_id}의 feature 생성 실패")
                return []
            
            # 6. 모델 예측
            feature_names = self._get_feature_names()
            X = features_df[feature_names]
            
            # 북마크할 확률 예측
            if hasattr(self.model, 'predict_proba'):
                prediction_scores = self.model.predict_proba(X)[:, 1]
            else:
                # LightGBM Booster인 경우
                prediction_scores = self.model.predict(X)
            
            # 7. 결과 정리 및 정렬
            features_df['prediction_score'] = prediction_scores
            top_recommendations = features_df.sort_values('prediction_score', ascending=False).head(limit * 2)  # 캐시용으로 더 많이 생성
            
            # 8. 추천 결과 형태로 변환
            recommendations = []
            for _, row in top_recommendations.iterrows():
                post_details = self._get_post_details(row['recruit_post_id'])
                
                recommendations.append({
                    "recruit_post_id": int(row['recruit_post_id']),
                    "score": float(row['prediction_score']),
                    "post_details": post_details,
                    "created_at": datetime.now().isoformat(),
                    "source": "realtime"
                })
            
            # 9. MongoDB에 캐시 저장 (10분 TTL)
            self._cache_realtime_recommendations(member_id, recommendations)
            
            logger.info(f"회원 {member_id} 실시간 추천 생성 완료: {len(recommendations)}개")
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"실시간 추천 생성 실패: {e}")
            return []
    
    def _get_cached_realtime_recommendations(self, member_id: int) -> Optional[List[Dict]]:
        """MongoDB에서 캐시된 실시간 추천 조회"""
        try:
            mongodb_manager = get_mongodb_manager()
            
            # realtime_cache 컬렉션에서 조회
            cache_doc = mongodb_manager.db.realtime_cache.find_one({
                "member_id": member_id,
                "expires_at": {"$gt": datetime.now()}  # 만료되지 않은 것만
            })
            
            if cache_doc:
                logger.debug(f"회원 {member_id} 캐시된 실시간 추천 발견")
                return cache_doc.get('recommendations', [])
            
            return None
            
        except Exception as e:
            logger.error(f"실시간 추천 캐시 조회 실패: {e}")
            return None
    
    def _cache_realtime_recommendations(self, member_id: int, recommendations: List[Dict]) -> bool:
        """MongoDB에 실시간 추천 캐시 저장"""
        try:
            mongodb_manager = get_mongodb_manager()
            
            # 10분 후 만료
            expires_at = datetime.now() + timedelta(minutes=10)
            
            cache_doc = {
                "member_id": member_id,
                "recommendations": recommendations,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "cache_type": "realtime_recommendations"
            }
            
            # 기존 캐시 삭제 후 새로 저장 (upsert)
            mongodb_manager.db.realtime_cache.replace_one(
                {"member_id": member_id},
                cache_doc,
                upsert=True
            )
            
            logger.debug(f"회원 {member_id} 실시간 추천 캐시 저장 완료 (만료: {expires_at})")
            return True
            
        except Exception as e:
            logger.error(f"실시간 추천 캐시 저장 실패: {e}")
            return False
    
    def _get_member_data(self, member_id: int) -> Optional[Dict]:
        """회원 데이터 조회"""
        try:
            postgres_client = get_postgresql_client()
            
            query = """
            SELECT
                m.id as member_id,
                m.gender as member_gender,
                m.birth_date,
                pp.lifestyle as profile_lifestyle,
                pp.personality as profile_personality,
                pp.is_smoking as profile_is_smoking,
                pp.is_snoring as profile_is_snoring,
                pp.has_pet as profile_has_pet,
                up.cohabitant_count,
                up.has_pet as pref_has_pet,
                up.is_smoking as pref_is_smoking,
                up.is_snoring as pref_is_snoring,
                up.gender as pref_gender,
                up.lifestyle as pref_lifestyle,
                up.personality as pref_personality
            FROM member m
            LEFT JOIN public_profile pp ON m.public_profile_id = pp.public_profile_id
            LEFT JOIN user_preferences up ON m.id = up.member_id
            WHERE m.id = %s AND m.is_completed = true
            """
            
            result = postgres_client.execute_query(query, (member_id,))
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"회원 데이터 조회 실패: {e}")
            return None
    
    def _get_active_posts_for_member(self, member_id: int) -> List[Dict]:
        """회원에게 추천할 수 있는 활성 구인글 조회 (자신의 구인글 제외)"""
        try:
            postgres_client = get_postgresql_client()
            
            query = """
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
                recruit_count,
                prefered_gender,
                personality,
                life_style,
                is_smoking,
                is_snoring,
                is_pets_allowed
            FROM recruit_post
            WHERE status = 'ACTIVE' 
            AND user_id != %s
            ORDER BY created_at DESC
            LIMIT 50
            """
            
            result = postgres_client.execute_query(query, (member_id,))
            return result if result else []
            
        except Exception as e:
            logger.error(f"활성 구인글 조회 실패: {e}")
            return []
    
    def _create_features_for_pairs(self, member_post_pairs: List[Dict]) -> pd.DataFrame:
        """회원-구인글 조합에 대한 feature 생성"""
        try:
            # 기존 feature_engineer를 활용하여 실시간 feature 생성
            # 간소화된 버전으로 구현
            
            features = []
            for pair in member_post_pairs:
                member_data = pair['member_data']
                post_data = pair['post_data']
                
                feature_row = {
                    'member_id': member_data['member_id'],
                    'recruit_post_id': post_data['post_id'],
                    # 기본 features
                    'age_match': self._calculate_age_match(member_data, post_data),
                    'gender_match': self._calculate_gender_match(member_data, post_data),
                    'lifestyle_match': self._calculate_lifestyle_match(member_data, post_data),
                    'personality_match': self._calculate_personality_match(member_data, post_data),
                    'smoking_match': self._calculate_smoking_match(member_data, post_data),
                    'pet_match': self._calculate_pet_match(member_data, post_data),
                    # 추가 features (간소화)
                    'is_own_post': 0,  # 이미 필터링됨
                    'is_bookmarked': 0  # 신규 회원이므로 북마크 없음
                }
                
                features.append(feature_row)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"실시간 feature 생성 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_age_match(self, member_data: Dict, post_data: Dict) -> float:
        """나이 매치 계산 (간소화)"""
        try:
            from datetime import date
            birth_date = member_data.get('birth_date')
            if not birth_date:
                return 0.5
            
            age = (date.today() - birth_date).days // 365
            min_age = post_data.get('min_age', 0)
            max_age = post_data.get('max_age', 100)
            
            if min_age <= age <= max_age:
                return 1.0
            else:
                return 0.0
        except:
            return 0.5
    
    def _calculate_gender_match(self, member_data: Dict, post_data: Dict) -> float:
        """성별 매치 계산"""
        member_gender = member_data.get('member_gender')
        preferred_gender = post_data.get('prefered_gender')
        
        if not preferred_gender or preferred_gender == 'ANY':
            return 1.0
        elif member_gender == preferred_gender:
            return 1.0
        else:
            return 0.0
    
    def _calculate_lifestyle_match(self, member_data: Dict, post_data: Dict) -> float:
        """라이프스타일 매치 계산"""
        member_lifestyle = member_data.get('profile_lifestyle')
        post_lifestyle = post_data.get('life_style')
        
        if not member_lifestyle or not post_lifestyle:
            return 0.5
        
        return 1.0 if member_lifestyle == post_lifestyle else 0.3
    
    def _calculate_personality_match(self, member_data: Dict, post_data: Dict) -> float:
        """성격 매치 계산"""
        member_personality = member_data.get('profile_personality')
        post_personality = post_data.get('personality')
        
        if not member_personality or not post_personality:
            return 0.5
        
        return 1.0 if member_personality == post_personality else 0.3
    
    def _calculate_smoking_match(self, member_data: Dict, post_data: Dict) -> float:
        """흡연 매치 계산"""
        member_smoking = member_data.get('profile_is_smoking', False)
        post_smoking_allowed = post_data.get('is_smoking', True)
        
        if post_smoking_allowed:
            return 1.0  # 흡연 허용하면 상관없음
        else:
            return 1.0 if not member_smoking else 0.0  # 흡연 금지인데 회원이 흡연자면 매치 안됨
    
    def _calculate_pet_match(self, member_data: Dict, post_data: Dict) -> float:
        """펫 매치 계산"""
        member_has_pet = member_data.get('profile_has_pet', False)
        post_pets_allowed = post_data.get('is_pets_allowed', True)
        
        if post_pets_allowed:
            return 1.0  # 펫 허용하면 상관없음
        else:
            return 1.0 if not member_has_pet else 0.0  # 펫 금지인데 회원이 펫 소유자면 매치 안됨
    
    def _get_feature_names(self) -> List[str]:
        """모델에서 사용하는 feature 이름들 반환"""
        # 실제로는 models/feature_names.txt에서 로드하거나
        # feature_engineer에서 가져와야 함
        return [
            'age_match', 'gender_match', 'lifestyle_match', 
            'personality_match', 'smoking_match', 'pet_match',
            'is_own_post', 'is_bookmarked'
        ]
    
    def _get_post_details(self, post_id: int) -> Dict:
        """구인글 상세 정보 조회"""
        try:
            postgres_client = get_postgresql_client()
            
            query = """
            SELECT title, address, has_room, monthly_cost_min, monthly_cost_max
            FROM recruit_post
            WHERE id = %s
            """
            
            result = postgres_client.execute_query(query, (post_id,))
            if result:
                post_data = result[0]
                return {
                    "title": post_data.get('title', ''),
                    "address": post_data.get('address', ''),
                    "has_room": post_data.get('has_room', False),
                    "monthly_cost_min": post_data.get('monthly_cost_min', 0),
                    "monthly_cost_max": post_data.get('monthly_cost_max', 0)
                }
            return {}
            
        except Exception as e:
            logger.error(f"구인글 상세 정보 조회 실패: {e}")
            return {}

# 싱글톤 패턴
_realtime_recommender_instance = None

def get_realtime_recommender() -> RealtimeRecommender:
    """RealtimeRecommender 싱글톤 반환"""
    global _realtime_recommender_instance
    if _realtime_recommender_instance is None:
        _realtime_recommender_instance = RealtimeRecommender()
    return _realtime_recommender_instance