"""data/raw/ 원본 CSV가 없으면 GitHub Release 자산에서 내려받는다.

Streamlit Community Cloud는 저장소만 클론하고 data/raw/*.csv는 없다(.gitignore로
제외, 용량 문제 — README 참고). 앱이 시작할 때 이 모듈로 파일 존재를 확인하고,
없으면 이 저장소의 GitHub Release(raw-data-v1)에 올려둔 tar.gz를 받아서 푼다.

로컬 개발 환경처럼 data/raw/에 파일이 이미 있으면 아무 것도 하지 않는다 — 재다운로드
없음, 원본 파일을 덮어쓰지 않음(CLAUDE.md 데이터 보존 규칙 1).

배포 데이터 자산 출처: dunnhumby - The Complete Journey (Kaggle), SOURCE_AND_LICENSE.md 참고.
자산 URL: https://github.com/hevmadle/stnew/releases/tag/raw-data-v1
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATA_ASSET_URL = (
    "https://github.com/hevmadle/stnew/releases/download/raw-data-v1/data_raw.tar.gz"
)

EXPECTED_FILES = [
    "campaign_desc.csv",
    "campaign_table.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "hh_demographic.csv",
    "product.csv",
    "transaction_data.csv",
]


def missing_files() -> list[str]:
    return [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]


def ensure_raw_data(url: str = DATA_ASSET_URL, progress_cb=None) -> bool:
    """data/raw/에 7개 파일이 모두 있으면 True를 반환하고 아무 것도 하지 않는다.

    하나라도 없으면 url의 tar.gz를 받아 data/raw/에 푼다(기존 파일은 건드리지 않음
    — tar 안에 있는 이름만 새로 생긴다). 성공하면 True, 실패하면 예외를 던진다.

    progress_cb: 선택적 콜백(str) — Streamlit st.spinner 등에서 진행 메시지를 보여줄 때 쓴다.
    """
    missing = missing_files()
    if not missing:
        return True

    def report(msg: str) -> None:
        print(msg)
        if progress_cb:
            progress_cb(msg)

    report(f"data/raw/에 없는 파일 {len(missing)}개 발견: {missing}")
    report(f"원본 데이터 내려받는 중... ({url})")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            with open(tmp_path, "wb") as out:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        report(f"  {downloaded/1e6:.1f} / {total/1e6:.1f} MB")

        report("압축 해제 중...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            # 이름이 EXPECTED_FILES 안에 있는 항목만 안전하게 추출 (경로 탈출 방지)
            safe_members = [m for m in tar.getmembers() if Path(m.name).name in EXPECTED_FILES]
            tar.extractall(path=RAW_DIR, members=safe_members, filter="data")
    finally:
        tmp_path.unlink(missing_ok=True)

    still_missing = missing_files()
    if still_missing:
        raise RuntimeError(f"다운로드 후에도 없는 파일: {still_missing}")

    report("완료: data/raw/ 준비됨")
    return True


if __name__ == "__main__":
    ensure_raw_data()
