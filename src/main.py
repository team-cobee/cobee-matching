"""
  FastAPI 메인 애플리케이션 - MongoDB 기반 추천 시스템
  - 실시간 추천: MongoDB에서 미리 계산된 추천 조회
  - 백그라운드: 30초 폴링으로 모델 업데이트 및 배치 추천 생성
"""
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import threading
import os

from .database.postgres_client import get_postgresql_client
from .database.mongodb_manager import get_mongodb_manager
from .ml.pipeline import get_ml_pipeline
from .ml.realtime_recommender import get_realtime_recommender
from .polling.scheduler import get_polling_scheduler
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_pipeline = None
postgres_client = None
mongodb_manager = None
polling_scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리
    """
    global ml_pipeline, postgres_client, mongodb_manager, polling_scheduler

    try:
        logger.info("애플리케이션 시작 - 리소스 초기화")
        postgres_client = get_postgresql_client()
        logger.info("PostgreSQL 클라이언트 초기화 완료")

        mongodb_manager = get_mongodb_manager()
        logger.info("MongoDB 매니저 초기화 완료")

        ml_pipeline = get_ml_pipeline()
        logger.info("ML 파이프라인 초기화 완료")

        polling_scheduler = get_polling_scheduler()
        logger.info("폴링 스케줄러 초기화 완료")

        def start_polling():
            polling_scheduler.start()
        polling_thread = threading.Thread(target=start_polling, daemon=True)
        polling_thread.start()

        logger.info("백그라운드 폴링 스레드 시작 완료")
    
        logger.info("애플리케이션 시작 완료")
        logger.info("- PostgreSQL 연결: ✅")
        logger.info("- MongoDB 연결: ✅")
        logger.info("- ML Pipeline: ✅")
        logger.info("- 폴링 스케줄러 (30초 주기): ✅")
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {e}")
        raise
    yield # yield 이전: 앱 시작 시 실행 (리소스 초기화) , yield 이후: 앱 종료 시 실행 (리소스 정리)

    try:
        if polling_scheduler:
            polling_scheduler.stop()
            logger.info("폴링 스케줄러 종료 완료")
        if postgres_client:
            postgres_client.disconnect()
            logger.info("PostgreSQL 연결 종료 완료")
        if mongodb_manager:
            mongodb_manager.disconnect()
            logger.info("MongoDB 연결 종료 완료")

        logger.info("✅ FastAPI 서버 종료")
    except Exception as e:
        logger.error(f"❌ 서버 종료 중 오류: {e}")

app = FastAPI(
    title="Cobee Matching API v2.0",
    description="""
    **MongoDB 기반 룸메이트 매칭 추천 시스템**
      
    ## 주요 특징
    - 🚀 실시간 추천: MongoDB에서 미리 계산된 추천 조회
    - 🔄 자동 업데이트: 30초마다 데이터 변경분 감지 및 모델 업데이트
    - 📊 배치 추천: 모델 업데이트 시 모든 사용자 추천 재생성
    - 📈 성능 기반 모델 교체: 새 모델이 기존보다 좋을 때만 교체
    """,
    version="2.0.0",
    contact={
        "name": "Cobee Matching Team",
    },
    lifespan=lifespan
)

@app.get("/", tags=["시스템"])
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Cobee Matching API v2.0",
        "status": "running",
        "recommendation_source": "mongodb_precomputed",
        "update_frequency": "30_seconds_polling",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["monitoring"])
async def health_check(): 
    """
    헬스체크 엔드포인트
    - PostgreSQL, MongoDB 연결 상태
    - ML 파이프라인 상태
    - 폴링 스케줄러 상태
    """
    try:
        postgres_status = postgres_client.health_check() if postgres_client else {"is_connected": False}

        mongodb_status = mongodb_manager.health_check() if mongodb_manager else {"mongodb_connected": False}

        ml_status = ml_pipeline.get_pipeline_status() if ml_pipeline else {"pipeline_status": "not_initialized"}

        polling_status = polling_scheduler.get_status() if polling_scheduler else {"is_running": False}

        overall_healthy = (
            postgres_status.get("is_connected", False) and
            mongodb_status.get("mongodb_connected", False) and
            ml_status.get("pipeline_status") != "error" and
            polling_status.get("is_running", False)
        )

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "postgres": postgres_status,
                "mongodb": mongodb_status,
                "ml_pipeline": ml_status,
                "polling_scheduler": polling_status
            },
            "version": "2.0.0"
          }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/recommend/{member_id}", tags=["추천"])
async def get_recommendations(
    member_id: int,
    limit: int = Query(4, ge=1, le=10, description="추천 개수 (1-10)"),
):
    """
    사용자별 룸메이트 매칭 추천 (MongoDB에서 미리 계산된 추천 조회 후 계층별 분배)
    
    호출 관계:
    - 내부 로직: 계층별 분배    
    """
    try:
        if not mongodb_manager:
            raise HTTPException(status_code=503, detail="MongoDB 매니저가 초기화되지 않았습니다.")

        # 1. MongoDB에서 해당 사용자의 모든 추천 결과 조회
        all_recommendations = mongodb_manager.get_member_recommendations(member_id)

        # 2. 추천 결과가 없는 경우 처리 - 실시간 추천 생성
        if not all_recommendations:
            logger.info(f"회원 {member_id} MongoDB 추천 결과 없음 - 실시간 추천 생성 시도")
            
            try:
                # 실시간 추천 생성기 사용
                realtime_recommender = get_realtime_recommender()
                realtime_recommendations = realtime_recommender.generate_realtime_recommendations(
                    member_id=member_id, 
                    limit=limit
                )
                
                if realtime_recommendations:
                    logger.info(f"회원 {member_id} 실시간 추천 생성 성공: {len(realtime_recommendations)}개")
                    return {
                        "member_id": member_id,
                        "recommendations": realtime_recommendations,
                        "count": len(realtime_recommendations),
                        "source": "realtime_generated",
                        "timestamp": datetime.now().isoformat(),
                        "message": "실시간으로 생성된 추천입니다."
                    }
                else:
                    logger.warning(f"회원 {member_id} 실시간 추천 생성 실패")
                    
            except Exception as e:
                logger.error(f"회원 {member_id} 실시간 추천 생성 중 오류: {e}")
            
            # 실시간 추천도 실패한 경우
            return {
                "member_id": member_id,
                "recommendations": [],
                "count": 0,
                "source": "fallback",
                "timestamp": datetime.now().isoformat(),
                "message": "현재 추천할 수 있는 구인글이 없습니다. 프로필을 완성하거나 잠시 후 다시 시도해주세요."
            }

        # 3. 임계값 가져오기
        all_scores = [r["score"] for r in all_recommendations]
        threshold = sum(all_scores) / len(all_scores) if all_scores else 0.5
        
        # 임계값 및 점수 분포 로그 출력
        logger.info(f"회원 {member_id} 추천 점수 분석:")
        logger.info(f"  - 전체 추천 개수: {len(all_recommendations)}")
        logger.info(f"  - 평균 점수 (임계값): {threshold:.6f}")
        logger.info(f"  - 최고 점수: {max(all_scores):.6f}")
        logger.info(f"  - 최저 점수: {min(all_scores):.6f}")

        # 4. 점수 기준으로 3개 클래스 분류
        high_class = [r for r in all_recommendations if r["score"] >= threshold + 0.2]
        mid_class = [r for r in all_recommendations if threshold + 0.1 <= r["score"] < threshold + 0.2]
        low_class = [r for r in all_recommendations if threshold <= r["score"] < threshold + 0.1]
        
        # 클래스별 분포 로그 출력
        logger.info(f"  - 고품질 클래스 (>= {threshold + 0.2:.6f}): {len(high_class)}개")
        logger.info(f"  - 중품질 클래스 ({threshold + 0.1:.6f} ~ {threshold + 0.2:.6f}): {len(mid_class)}개")
        logger.info(f"  - 저품질 클래스 ({threshold:.6f} ~ {threshold + 0.1:.6f}): {len(low_class)}개")
        logger.info(f"  - 임계값 미만 (<{threshold:.6f}): {len(all_recommendations) - len(high_class) - len(mid_class) - len(low_class)}개")

        # 5. 각 클래스별 할당량 계산
        high_quota = int(limit * 0.5)  # 50%
        mid_quota = int(limit * 0.25)  # 25% 
        # base_quota = limit - high_quota - mid_quota  # 나머지 25% (현재 미사용)

        # TODO 6: 고품질 클래스에서 우선 선택 (해당 클래스에서 랜덤으로 선택)
        final_recommendations = []
        if high_class:
            selected_high = random.sample(high_class, min(high_quota, len(high_class)))
            final_recommendations.extend(selected_high)
        # TODO 7: 중품질 클래스에서 선택 (고품질 부족분 보충 포함 + 랜덤 선택)
        remaining_quota = limit - len(final_recommendations)
        if mid_class and remaining_quota > 0:
            selected_mid = random.sample(mid_class, min(remaining_quota, len(mid_class)))
            final_recommendations.extend(selected_mid)
        # TODO 8: 저품질 클래스에서 부족분 보충 (랜덤 선택)
        remaining_quota = limit - len(final_recommendations)
        if low_class and remaining_quota > 0:
            selected_low = random.sample(low_class, min(remaining_quota, len(low_class)))
            final_recommendations.extend(selected_low)
        # TODO 9: 최종 결과 반환 형태 구성 (결과 셔플링 후 순서 없이 반환)
        random.shuffle(final_recommendations)
        return {
            "member_id": member_id,
            "recommendations": final_recommendations, # TODO: 실제 추천 결과로 변경 : final_recommendations
            "count": len(final_recommendations), # TODO: 실제 추천 결과 개수로 변경 : len(final_recommendations)
            "source": "mongodb_precomputed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"추천 조회 실패 - member_id: {member_id}, error: {e}")
        raise HTTPException(status_code=500, detail=f"추천 조회 실패: {str(e)}")
    
@app.post("/ml/train", tags=["ML"])
async def trigger_ml_training(force_full_training: bool = False):
    """
    ML 모델 재학습 트리거 (수동)
    """
    try:
        if not ml_pipeline:
            raise HTTPException(status_code=503, detail="ML 파이프라인이 초기화되지 않았습니다.")
        logger.info("수동 모델 학습 실행")

        # ML 파이프라인에서 모델 학습 실행
        result = ml_pipeline.train_model(force_retrain=force_full_training)

        return {
            "training_triggered": True,
            "training_id": result.get("training_id"),
            "status": result.get("status"),
            "model_version": result.get("model_version"),
            "metrics": result.get("metrics"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"ML 학습 트리거 실패: {e}")
        raise HTTPException(status_code=500, detail=f"ML 모델 학습 실패: {str(e)}")
    
@app.get("/ml/status", tags=["ML"])
async def get_ml_status():
    """
    ML 파이프라인 상태 조회

    호출 관계:
    - ml_pipeline.get_pipeline_status() → ML 파이프라인 상태 정보
    - polling_scheduler.get_status() → 폴링 스케줄러 상태 정보
    """
    try:
        if not ml_pipeline:
            raise HTTPException(status_code=503, detail="ML 파이프라인이 초기화되지 않았습니다.")

        # ML 파이프라인에서 모델 상태 조회
        ml_status = ml_pipeline.get_pipeline_status()
        polling_status = polling_scheduler.get_status() if polling_scheduler else {"is_running": False}

        return {
            "ml_pipeline": ml_status,
            "polling_scheduler": polling_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"ML 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"ML 상태 조회 실패: {str(e)}")
    
@app.get("/ml/training-logs", tags=["ML"])
async def get_ml_training_logs(limit: int = Query(10, ge=1, le=100, description="로그 개수 (1-100)")):
    """
    ML 모델 학습 로그 조회
   """
    try:
        if not mongodb_manager:
            raise HTTPException(status_code=503, detail="MongoDB 매니저가 초기화되지 않았습니다.")
        
        training_logs = mongodb_manager.get_recent_training_logs(limit)

        return {
            "training_logs": training_logs,
            "count": len(training_logs),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"학습 로그 조회 실패 : {e}")
        raise HTTPException(status_code=500, detail=f"학습 로그 조회 실패: {str(e)}")


@app.get("/members/{member_id}/recommendations/history", 
tags=["사용자"])
async def get_recommendation_history(
    member_id: int,
    limit: int = Query(10, ge=1, le=50, description="추천 기록 개수")
):
    """
      사용자 추천 기록 조회
    """
    try:
        if not mongodb_manager:
            raise HTTPException(status_code=503, detail="MongoDB 매니저가 초기화되지 않았습니다.")

        # MongoDB에서 해당 사용자의 추천 기록 조회
        recommendation_history = mongodb_manager.get_member_recommendation_history(member_id, limit)

        return {
            "member_id": member_id,
            "recommendation_history": recommendation_history,
            "count": len(recommendation_history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"회원 {member_id} 추천 기록 조회 실패 : {e}")
        raise HTTPException(status_code=500, detail=f"추천 기록 조회 실패: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)