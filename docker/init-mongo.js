// MongoDB 초기화 스크립트 - data_generator.py 스키마에 정확히 맞춤
// Docker 컨테이너 시작 시 자동 실행됨

print('🍃 Initializing roommate_matching database...');

// 데이터베이스 선택
db = db.getSiblingDB('roommate_matching');

// 애플리케이션 사용자 생성
db.createUser({
  user: process.env.MONGODB_APP_USER || 'default_user',
  pwd: process.env.MONGODB_APP_PASSWORD || 'default_password',
  roles: [{ role: 'readWrite', db: 'roommate_matching' }]
});
