"""
    폴링 스케줄러
"""
import threading
import time
import logging
from datetime import datetime
from typing import Optional
from ..database.postgres_client import get_postgresql_client
from ..ml.pipeline import get_ml_pipeline

logger = logging.getLogger(__name__)

class PostgreSQLPollingScheduler:
    def __init__(self):
        """폴링 스케줄러 초기화"""
        self.is_running = False
        self.poll_interval = 30 # 30초
        self.last_check = None
        self.polling_thread = None
        self._stop_event = threading.Event()
        logger.info("폴링 스케줄러 초기화 완료")

    def start(self) -> None:
        """30초 주기 폴링 시작"""
        if self.is_running:
            logger.warning("폴링 스케줄러가 이미 실행중입니다.")
            return
        self.is_running = True
        self._stop_event.clear() # 질문 : 이게 무슨 기능인지? 어디에 있는지?
        # _polling_loop()를 백그라운드에서 주기적으로 실행
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("폴링 스케줄러 시작 완료")
        
    def stop(self) -> None:
        """폴링 중지"""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()

        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)

    def get_status(self) -> dict:
        """폴링 상태 조회"""
        # 반환: {"is_running": bool, "last_check": datetime, "poll_interval": int}
        return {
            "is_running": self.is_running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "poll_interval": self.poll_interval
        }
    
    def _polling_loop(self) -> None:
        """폴링 작업 수행"""
        # stop 신호가 오지 않은 동안 계속 실행
        while not self._stop_event.is_set(): # self._stop_event : 객체로, 스레드 간 신호를 주고받는 도구
            try:
                self.last_check = datetime.now()
                logger.info(f"폴링 시작: {self.last_check}")
                
                # 1. 데이터 변경분 확인
                changes_result = self._check_data_changes()
                if changes_result.get("has_changes", False):
                    logger.info(f"데이터 변경분 감지: {changes_result['total_changes']}개")
                    # 2. ML 파이프라인 트리거
                    self._trigger_ml_pipeline()
                else:
                    logger.info("데이터 변경분 없음 - ML 파이프라인 건너뛰기")
            except Exception as e:
                logger.error(f"폴링 루프 오류: {e}")
                self._stop_event.wait(self.poll_interval)
            self._stop_event.wait(self.poll_interval)

    def _check_data_changes(self) -> dict:
        """데이터베이스 변경분 감지"""
        try:
            logger.info("🔍 데이터 변경분 확인 시작")
            # postgres_client 호출
            postgres_client = get_postgresql_client()
            result =  postgres_client.check_data_changes()
            logger.info(f"🔍 변경분 확인 결과: {result}")
            return result
        except Exception as e:
            logger.error(f"데이터 변경분 확인 실패 : {e}")
            return {
                "has_changes": False,
                "error": str(e),
                "check_time": datetime.now().isoformat()                
            }
        
    def _trigger_ml_pipeline(self) -> None:
        """ML 파이프라인 트리거"""
        # 호출 관계: ml_pipeline.handle_data_changes()
        try:
            ml_pipeline = get_ml_pipeline()
            logger.info("ML 파이프라인 시작")

            # 데이터 변경분 처리 시작
            result = ml_pipeline.handle_data_changes()
            logger.info(f"ML 파이프라인 완료: {result}")
        except Exception as e:
            logger.error(f"ML 파이프라인 실행 실패: {e}")

_polling_scheduler_instance = None

def get_polling_scheduler() -> PostgreSQLPollingScheduler:
    """폴링 스케줄러 싱글톤 반환"""
    global _polling_scheduler_instance
    if _polling_scheduler_instance is None:
        _polling_scheduler_instance = PostgreSQLPollingScheduler()
    return _polling_scheduler_instance