import pandas as pd
from typing import List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.members_data = None
        self.recruit_posts_data = None
        self.bookmarks_data = None
        self.user_preferences_data = None
        self.public_profile_data = None

    def process_full_data(self, data: dict) -> dict:
        """전체 데이터에 대한 특성 엔지니어링 수행"""
        # List를 pandas DataFrame으로 변환
        self.members_data = pd.DataFrame(data.get('members', []))
        self.recruit_posts_data = pd.DataFrame(data.get('recruit_posts', []))
        self.bookmarks_data = pd.DataFrame(data.get('bookmarks', []))
        self.user_preferences_data = pd.DataFrame(data.get('user_preferences', []))
        self.public_profile_data = pd.DataFrame(data.get('public_profile', []))
        
        
        # 모든 회원과 구인글의 가능한 조합을 만드는 메서드 호출
        pairs_df = self._create_all_pairs()
        
        # 각 회원-구인글 조합에 대해 기계학습 특성들을 생성하는 메서드 호출
        features_df = self._create_features_for_pairs(pairs_df)
        
        # 생성된 특성들에 정답 라벨(북마크 여부)을 추가하는 메서드 호출
        features_with_labels = self._add_labels(features_df)
        
        return {
            # 완성된 특성 데이터(features + labels) 반환
            'features': features_with_labels,
            # 특성 이름들의 리스트 반환 (모델 학습 시 필요)
            'feature_names': self._get_feature_names(features_with_labels)
        }

    def _create_all_pairs(self) -> pd.DataFrame:
        """모든 member-recruit_post 조합 생성"""
        if self.members_data.empty or self.recruit_posts_data.empty:
            return pd.DataFrame()
        
        pairs = []
        for member_id in self.members_data['member_id']:
            for post_id in self.recruit_posts_data['post_id']:
                pairs.append({
                    'member_id': member_id,
                    'recruit_post_id': post_id
                })
        
        return pd.DataFrame(pairs)

    def _create_features_for_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """각 member-post 조합에 대한 features 생성"""
        features_list = []
        
        for _, row in pairs_df.iterrows():
            member_id = row['member_id']
            post_id = row['recruit_post_id']
            
            member = self._get_member_data(member_id)
            post = self._get_post_data(post_id)
            # 회원 데이터에 이미 JOIN된 정보 사용 (별도 조회 불필요)
            user_pref = member  # user_preferences는 이미 member에 JOIN되어 있음
            public_profile = member  # public_profile도 이미 member에 JOIN되어 있음
            
            features = {}
            features['member_id'] = member_id
            features['recruit_post_id'] = post_id
            
            # 각 카테고리별 features 생성 주석
            basic_features = self._create_basic_compatibility_features(member, post, user_pref, public_profile)
            # 기본 호환성 특성들을 메인 특성 딕셔너리에 병합
            features.update(basic_features)
            features.update(self._create_cost_features(post))
            features.update(self._create_post_features(post, member_id))
            features.update(self._create_temporal_features(member, post))
            features.update(self._create_statistical_features(member_id, post_id))
            features.update(self._create_composite_features(basic_features))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def _get_member_data(self, member_id: int) -> dict:
        """특정 member 데이터 조회"""
        # 회원 데이터에서 지정된 ID와 일치하는 행 필터링
        member_row = self.members_data[self.members_data['member_id'] == member_id]
        # 조회된 행이 있으면 첫 번째 행을 딕셔너리로 변환, 없으면 빈 딕셔너리 반환
        return member_row.iloc[0].to_dict() if not member_row.empty else {}
    
    def _get_post_data(self, post_id: int) -> dict:
        """특정 recruit_post 데이터 조회"""
        post_row = self.recruit_posts_data[self.recruit_posts_data['post_id'] == post_id]
        return post_row.iloc[0].to_dict() if not post_row.empty else {}
    
    def _get_user_preferences(self, member_id: int) -> dict:
        """특정 member의 user_preferences 데이터 조회"""
        # 사용자 선호도 데이터에서 지정된 회원 ID와 일치하는 행 필터링
        pref_row = self.user_preferences_data[self.user_preferences_data['member_id'] == member_id]
        # 조회된 행이 있으면 첫 번째 행을 딕셔너리로 변환, 없으면 빈 딕셔너리 반환
        return pref_row.iloc[0].to_dict() if not pref_row.empty else {}
    
    # 특정 공개 프로필 데이터를 조회하는 메서드
    def _get_public_profile(self, profile_id) -> dict:
        """특정 public_profile 데이터 조회"""
        # 프로필 ID가 비어있거나 공개 프로필 데이터가 없는 경우 체크
        if pd.isna(profile_id) or self.public_profile_data.empty:
            # 빈 딕셔너리 반환
            return {}
        # 공개 프로필 데이터에서 지정된 프로필 ID와 일치하는 행 필터링
        profile_row = self.public_profile_data[self.public_profile_data['public_profile_id'] == profile_id]
        # 조회된 행이 있으면 첫 번째 행을 딕셔너리로 변환, 없으면 빈 딕셔너리 반환
        return profile_row.iloc[0].to_dict() if not profile_row.empty else {}

    # 회원과 구인글 간의 기본적인 호환성을 계산하는 메서드
    def _create_basic_compatibility_features(self, member: dict, post: dict, _user_pref: dict, public_profile: dict) -> dict:
        """기본 호환성 features 생성"""
        features = {}
        
        # 1. 나이 호환성 주석
        # 회원의 생년월일로부터 나이 계산
        member_age = self._calculate_age(member.get('birth_date'))
        # 구인글에서 요구하는 최소 나이 (기본값 19)
        post_min_age = post.get('min_age', 19)
        # 구인글에서 요구하는 최대 나이 (기본값 100)
        post_max_age = post.get('max_age', 100)
        
        if member_age and post_min_age and post_max_age:
            # 회원 나이가 구인글 나이 범위에 포함되면 1, 아니면 0
            features['age_in_range'] = 1 if post_min_age <= member_age <= post_max_age else 0
            # 회원 나이와 최소 나이의 차이 계산
            features['age_distance_min'] = abs(member_age - post_min_age)
            # 회원 나이와 최대 나이의 차이 계산
            features['age_distance_max'] = abs(member_age - post_max_age)
        else:
            features['age_in_range'] = 0
            features['age_distance_min'] = 0
            features['age_distance_max'] = 0
        
        # 2. 성별 호환성 주석
        # 회원의 성별 정보 (기본값 'NONE')
        # 참고 : 회원의 성별은 MALE, FEMALE 만 존재, NONE은 성별 정보가 없다는 뜻 - 하지만 우리 서비스에서는 성별 정보를 제공하지 않으면 가입 불가
        member_gender = member.get('gender', 'NONE')
        # 구인글에서 선호하는 성별 (기본값 'NONE')
        post_preferred_gender = post.get('prefered_gender', 'NONE')
        # 구인글이 성별 무관이거나 회원 성별과 일치하면 1, 아니면 0
        features['gender_match'] = 1 if post_preferred_gender in ['NONE', member_gender] else 0
        
        # 3. 반려동물 호환성 주석
        # 회원 프로필의 반려동물 보유 여부 (기본값 'NONE')
        profile_has_pet = public_profile.get('has_pet', 'NONE')
        # 구인글의 반려동물 허용 여부 (기본값 'NONE')
        post_pets_allowed = post.get('is_pets_allowed', 'NONE')
        # 구인글이 반려동물을 허용하거나 무관하면 1, 아니면 0
        features['pets_compatible'] = 1 if post_pets_allowed in ['NONE', 'POSSIBLE'] or profile_has_pet == 'NONE' else 0
        
        # 4. 흡연 호환성 주석
        # 회원 프로필의 흡연 여부 (기본값 'NONE')
        profile_smoking = public_profile.get('is_smoking', 'NONE')
        # 구인글의 흡연 허용 여부 (기본값 'NONE')
        post_smoking = post.get('is_smoking', 'NONE')
        # 구인글이 흡연 무관이거나 회원이 흡연 무관이면 1, 아니면 0
        features['smoking_compatible'] = 1 if post_smoking == 'NONE' or profile_smoking == 'NONE' else 0
        
        # 5. 코골이 호환성 주석
        profile_snoring = public_profile.get('is_snoring', 'NONE')
        post_snoring = post.get('is_snoring', 'NONE')
        features['snoring_compatible'] = 1 if post_snoring == 'NONE' or profile_snoring == 'NONE' else 0
        
        # 6. 생활패턴 호환성 주석
        profile_lifestyle = public_profile.get('lifestyle', 'NONE')
        post_lifestyle = post.get('life_style', 'NONE')
        features['lifestyle_match'] = 1 if post_lifestyle in ['NONE', profile_lifestyle] else 0
        
        # 7. 성격 호환성 주석
        profile_personality = public_profile.get('personality', 'NONE')
        post_personality = post.get('personality', 'NONE')
        features['personality_match'] = 1 if post_personality in ['NONE', profile_personality] else 0

        return features

    # 구인글의 비용 관련 특성들을 생성하는 메서드
    def _create_cost_features(self, post: dict) -> dict:
        """비용 관련 features 생성"""
        features = {}
        
        # 월 비용 주석
        monthly_min = post.get('monthly_cost_min', 0) or 0
        monthly_max = post.get('monthly_cost_max', 0) or 0
        features['monthly_cost_avg'] = (monthly_min + monthly_max) / 2 if monthly_max > 0 else monthly_min
        features['monthly_cost_range'] = monthly_max - monthly_min
        features['monthly_cost_min'] = monthly_min
        features['monthly_cost_max'] = monthly_max
        
        # 보증금 주석
        rent_min = post.get('rent_cost_min', 0) or 0
        rent_max = post.get('rent_cost_max', 0) or 0
        features['rent_cost_avg'] = (rent_min + rent_max) / 2 if rent_max > 0 else rent_min
        features['rent_cost_range'] = rent_max - rent_min
        features['rent_cost_min'] = rent_min
        features['rent_cost_max'] = rent_max
        
        return features

    # 구인글의 기본 특성들을 생성하는 메서드
    def _create_post_features(self, post: dict, member_id: int) -> dict:
        """구인글 특성 features 생성"""
        features = {}
        
        # 모집 인원수 (기본값 2)
        features['recruit_count'] = post.get('recruit_count', 2)
        features['has_room'] = 1 if post.get('has_room', False) else 0
        # 지역의 위도 (기본값: 서울 위도 37.5665, None이면 기본값으로 처리)
        features['region_latitude'] = post.get('region_latitude') or 37.5665
        # 지역의 경도 (기본값: 서울 경도 126.9780, None이면 기본값으로 처리)
        features['region_longitude'] = post.get('region_longitude') or 126.9780
        features['is_seoul'] = 1 if '서울' in str(post.get('address', '')) else 0
        
        # 자신의 구인글 여부 feature 추가
        # 구인글 작성자 ID와 현재 회원 ID 비교
        post_author_id = post.get('user_id')
        features['is_own_post'] = 1 if post_author_id == member_id else 0
        
        return features

    # 시간과 관련된 특성들을 생성하는 메서드
    def _create_temporal_features(self, member: dict, post: dict) -> dict:
        """시간적 features 생성"""
        features = {}
        
        # 구인글 시간 정보 주석
        post_created = post.get('created_at')
        # 생성 일시 정보가 있는 경우
        if post_created:
            # 문자열 형태면 datetime 객체로 변환
            if isinstance(post_created, str):
                post_created = pd.to_datetime(post_created)
            # 현재 시간에서 구인글 생성 시간을 빼서 경과 일수 계산
            features['post_age_days'] = (datetime.now() - post_created).days
            # 구인글이 생성된 요일 (0=월요일, 6=일요일)
            features['post_day_of_week'] = post_created.weekday()
            # 구인글이 생성된 월 (1-12)
            features['post_month'] = post_created.month
        else:
            features['post_age_days'] = 0
            features['post_day_of_week'] = 0
            features['post_month'] = 1
        
        # 회원 시간 정보 주석
        # 회원 가입 일시 정보 가져오기
        member_created = member.get('created_at')
        # 가입 일시 정보가 있는 경우
        if member_created:
            # 문자열 형태면 datetime 객체로 변환
            if isinstance(member_created, str):
                member_created = pd.to_datetime(member_created)
            features['member_age_days'] = (datetime.now() - member_created).days
            # 회원이 가입한 월 (1-12)
            features['member_registration_month'] = member_created.month
        # 가입 일시 정보가 없는 경우 기본값 설정
        else:
            features['member_age_days'] = 0
            features['member_registration_month'] = 1
        
        return features

    # 통계를 기반으로 한 특성들을 생성하는 메서드
    def _create_statistical_features(self, member_id: int, post_id: int) -> dict:
        """통계 기반 features 생성"""
        features = {}
        
        # 사용자 북마크 통계 주석
        # 해당 회원이 북마크한 모든 구인글 필터링
        user_bookmarks = self.bookmarks_data[self.bookmarks_data['member_id'] == member_id]
        # 해당 회원의 총 북마크 수
        features['user_total_bookmarks'] = len(user_bookmarks)
        
        # 구인글 인기도 통계 주석
        # 해당 구인글을 북마크한 모든 회원 필터링
        post_bookmarks = self.bookmarks_data[self.bookmarks_data['recruit_post_id'] == post_id]
        # 해당 구인글이 받은 총 북마크 수
        features['post_bookmark_count'] = len(post_bookmarks)
        
        # 전체 통계 대비 비율 주석
        # 전체 회원 수 계산
        total_members = len(self.members_data)
        # 전체 구인글 수 계산
        total_posts = len(self.recruit_posts_data)
        
        # 회원의 북마크 활동성 = 총 북마크 수 / 전체 구인글 수
        features['user_bookmark_rate'] = features['user_total_bookmarks'] / total_posts if total_posts > 0 else 0
        # 구인글의 인기도 점수 = 받은 북마크 수 / 전체 회원 수
        features['post_popularity_score'] = features['post_bookmark_count'] / total_members if total_members > 0 else 0
        
        return features

    # 여러 기본 특성들을 조합한 종합 점수 특성들을 생성하는 메서드
    def _create_composite_features(self, basic_features: dict) -> dict:
        """종합 점수 features 생성"""
        # 특성들을 저장할 빈 딕셔너리 생성
        features = {}
        
        # 기본 호환성 요소들 주석
        # 종합 점수 계산에 사용할 기본 호환성 항목들의 리스트
        compatibility_items = [
            'age_in_range', 'gender_match', 'pets_compatible', 
            'smoking_compatible', 'snoring_compatible', 
            'lifestyle_match', 'personality_match'
        ]
        
        # 매칭되는 조건 개수 주석
        # 기본 호환성 항목들 중 매칭되는 것들의 총합 계산
        match_count = sum(basic_features.get(item, 0) for item in compatibility_items)
        # 매칭된 조건의 개수를 특성으로 추가
        features['preference_match_count'] = match_count
        
        # 기본 호환성 점수 (평균) 주석
        # 매칭된 조건 수를 전체 조건 수로 나눈 평균 점수
        features['basic_compatibility_score'] = match_count / len(compatibility_items)
        
        # 수동 가중치 제거: LightGBM이 개별 호환성 feature들을 자동으로 학습하도록 함
        # 각 호환성 feature (age_in_range, gender_match 등)를 개별적으로 제공하여
        # 모델이 데이터에서 최적의 조합과 중요도를 스스로 학습
        
        # 생성된 종합 점수 특성들을 반환
        return features

    # 생년월일 문자열로부터 나이를 계산하는 메서드
    # 참고사항: member에서 생년월일은 YYYYMMDD 또는 YYYY-MM-DD 형태로 저장됨
    def _calculate_age(self, birth_date) -> int:
        """생년월일로부터 나이 계산"""
        # 생년월일이 비어있거나 None인 경우 체크
        if pd.isna(birth_date) or not birth_date:
            # 기본값으로 25세 반환
            return 25  # 기본값
        
        # 나이 계산을 시도하되, 에러 발생 시 기본값 반환
        try:
            # 생년월일이 문자열인 경우 형태별로 처리
            if isinstance(birth_date, str):
                birth_date = birth_date.strip()  # 공백 제거
                
                # YYYYMMDD 형태 (예: "19950315")
                if len(birth_date) == 8 and birth_date.isdigit():
                    # YYYY-MM-DD 형태로 변환하여 파싱
                    formatted_date = f"{birth_date[:4]}-{birth_date[4:6]}-{birth_date[6:8]}"
                    birth_date = pd.to_datetime(formatted_date)
                # YYYY-MM-DD 형태 또는 기타 표준 형태
                else:
                    birth_date = pd.to_datetime(birth_date)
            
            # 현재 년도에서 출생 년도를 빼서 나이 계산
            current_year = datetime.now().year
            birth_year = birth_date.year
            
            # 기본 나이 계산
            age = current_year - birth_year
            
            # 생일이 지났는지 확인하여 정확한 나이 계산
            current_date = datetime.now()
            if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
                age -= 1  # 생일이 아직 안 지났으면 1살 빼기
            
            return age
            
        # 변환이나 계산에서 에러 발생 시
        except Exception as e:
            # 디버깅을 위해 에러 정보 출력 (실제 운영에서는 로깅으로 대체)
            print(f"생년월일 파싱 에러: {birth_date}, 에러: {e}")
            # 기본값으로 25세 반환
            return 25

    # 생성된 특성 데이터에 정답 라벨(북마크 여부)을 추가하는 메서드
    def _add_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """북마크 여부를 라벨로 추가"""
        bookmark_set = set()
        # 모든 북마크 데이터를 순회
        for _, bookmark in self.bookmarks_data.iterrows():
            # (회원ID, 구인글ID) 튜플을 집합에 추가
            bookmark_set.add((bookmark['member_id'], bookmark['recruit_post_id']))
        
        features_df['is_bookmarked'] = features_df.apply(
            # 현재 행의 (회원ID, 구인글ID)가 북마크 집합에 있으면 1, 없으면 0
            lambda row: 1 if (row['member_id'], row['recruit_post_id']) in bookmark_set else 0,
            axis=1
        )
        
        return features_df

    # 기계학습에서 사용할 특성 이름들의 리스트를 반환하는 메서드
    def _get_feature_names(self, features_df: pd.DataFrame) -> List[str]:
        """feature 이름 목록 반환"""
        return [col for col in features_df.columns 
                if col not in ['member_id', 'recruit_post_id', 'is_bookmarked']]

# 싱글톤 패턴을 위한 전역 변수 - FeatureEngineer 인스턴스를 하나만 유지
_feature_engineer_instance = None
# FeatureEngineer 객체를 싱글톤으로 반환하는 함수
def get_feature_engineer() -> FeatureEngineer:
    """FeatureEngineer 싱글톤 반환"""
    # 전역 변수 사용 선언
    global _feature_engineer_instance
    # 인스턴스가 아직 생성되지 않았으면
    if _feature_engineer_instance is None:
        # 새로운 FeatureEngineer 인스턴스 생성
        _feature_engineer_instance = FeatureEngineer()
    # 기존 또는 새로 생성된 인스턴스 반환
    return _feature_engineer_instance