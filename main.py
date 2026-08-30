import base64
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 공통 설정
# =========================================================
KST = timezone(timedelta(hours=9))

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]
LLM_VALIDATION_RETRIES = 3

HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
DEFAULT_MAIN_HEX = "#12304A"
DEFAULT_SUB_HEXES = ["#65C7D0", "#F5F8F7", "#B8C4CC"]


# =========================================================
# OpenAI 설정
# =========================================================
def get_openai_client():
    """환경변수 OPENAI_API_KEY로 OpenAI 클라이언트를 생성합니다."""

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            "프로젝트 최상위 .env 파일을 확인해주세요."
        )

    return OpenAI(api_key=api_key)


# =========================================================
# 결과 구조 / 오류 기록 / 메타데이터
# =========================================================
def build_brand_result():
    """신규 실행용 brand_result 기본 구조를 만듭니다."""

    return {
        "brand_names": [],
        "slogans": [],
        "brand_story": "",
        "colors": {},
        "logos": [],
        "metadata": {
            "status": "success",
            "saved_at": "",
            "output_directory": "",
            "brand_name_count": 0,
            "slogan_count": 0,
            "logo_success_count": 0,
        },
        "errors": [],
    }


def add_error(
    brand_result,
    stage,
    item,
    message,
    attempts=1,
):
    """단계별 최종 실패를 errors 배열에 기록합니다."""

    # 같은 단계/항목의 기존 오류가 있다면 최신 내용으로 교체
    brand_result.setdefault("errors", [])
    brand_result["errors"] = [
        error
        for error in brand_result["errors"]
        if not (
            error.get("stage") == stage
            and error.get("item") == item
        )
    ]

    brand_result["errors"].append(
        {
            "stage": stage,
            "item": item,
            "message": str(message),
            "attempts": attempts,
        }
    )


def remove_error(brand_result, stage, item):
    """재시도 후 성공한 항목의 과거 오류 기록을 제거합니다."""

    brand_result.setdefault("errors", [])
    brand_result["errors"] = [
        error
        for error in brand_result["errors"]
        if not (
            error.get("stage") == stage
            and error.get("item") == item
        )
    ]


def _has_valid_core_result(brand_result):
    """핵심 브랜드 결과가 유효한지 확인합니다."""

    return not validate_llm_result(
        {
            "brand_names": brand_result.get("brand_names", []),
            "slogans": brand_result.get("slogans", []),
            "brand_story": brand_result.get("brand_story", ""),
            "colors": brand_result.get("colors", {}),
        }
    )


def portable_output_directory(output_directory):
    """
    결과 JSON에 개발 PC/샌드박스의 절대경로가 남지 않도록
    가능한 경우 output/... 형태의 상대경로로 기록합니다.
    """

    path = Path(output_directory)
    parts = list(path.parts)

    if "output" in parts:
        index = parts.index("output")
        return Path(*parts[index:]).as_posix()

    return path.as_posix()


def finalize_metadata(brand_result, output_directory):
    """저장 직전에 실행 상태와 결과 개수를 계산합니다."""

    brand_result.setdefault("metadata", {})
    brand_result.setdefault("errors", [])
    brand_result.setdefault("logos", [])

    metadata = brand_result["metadata"]

    metadata["saved_at"] = datetime.now(
        KST
    ).isoformat(timespec="seconds")

    metadata["output_directory"] = portable_output_directory(
        output_directory
    )

    metadata["brand_name_count"] = len(
        brand_result.get("brand_names", [])
    )

    metadata["slogan_count"] = len(
        brand_result.get("slogans", [])
    )

    metadata["logo_success_count"] = len(
        brand_result.get("logos", [])
    )

    # 핵심 브랜드 결과 자체를 얻지 못한 경우 failed
    if not _has_valid_core_result(brand_result):
        metadata["status"] = "failed"

    # 핵심 결과는 있으나 일부 최종 오류가 남은 경우 partial
    elif brand_result["errors"]:
        metadata["status"] = "partial"

    else:
        metadata["status"] = "success"


def print_summary(brand_result):
    """최종 실행 결과를 터미널에 간단히 요약합니다."""

    metadata = brand_result.get("metadata", {})
    colors = brand_result.get("colors", {})
    errors = brand_result.get("errors", [])

    main_count = 1 if colors.get("main") else 0
    sub_count = len(colors.get("sub", []))

    logo_fail_count = sum(
        1
        for error in errors
        if error.get("stage") == "logo_generation"
    )

    logo_success_count = metadata.get(
        "logo_success_count",
        len(brand_result.get("logos", [])),
    )

    logo_attempt_count = (
        logo_success_count + logo_fail_count
    )

    print("\n[생성 결과 요약]")
    print(
        "  브랜드명 후보 : "
        f"{metadata.get('brand_name_count', 0)}개"
    )
    print(
        "  슬로건        : "
        f"{metadata.get('slogan_count', 0)}개"
    )
    print(
        "  컬러          : "
        f"메인 {main_count}개 + 서브 {sub_count}개"
    )

    if logo_attempt_count:
        print(
            "  로고          : "
            f"{logo_success_count}/{logo_attempt_count} 성공"
        )
    else:
        print(
            "  로고          : 생성하지 않음"
        )

    print(
        f"  오류          : {len(errors)}건"
    )
    print(
        "  상태          : "
        f"{metadata.get('status', 'unknown')}"
    )


# =========================================================
# 기존 brand_result.json 자동 보완
# =========================================================
def migrate_brand_result(
    brand_result,
    result_path,
):
    """
    구버전 brand_result.json에 metadata/errors/logos가 없으면
    API 재호출 없이 최신 구조로 보완합니다.
    """

    changed = False
    output_dir = result_path.parent

    if "errors" not in brand_result:
        brand_result["errors"] = []
        print("  errors 없음 → 자동 추가")
        changed = True

    if "logos" not in brand_result:
        # 기존 PNG가 있으면 성공한 로고로 자동 인식
        existing_logos = []
        for index in range(1, 4):
            path = output_dir / f"logo_{index:02d}.png"
            if path.exists():
                existing_logos.append(
                    path.name
                )

        brand_result["logos"] = existing_logos
        print(
            "  logos 없음 → 기존 로고 파일 기준 자동 추가"
        )
        changed = True

    if "metadata" not in brand_result:
        brand_result["metadata"] = {
            "status": "success",
            "saved_at": "",
            "output_directory": portable_output_directory(output_dir),
            "brand_name_count": len(
                brand_result.get("brand_names", [])
            ),
            "slogan_count": len(
                brand_result.get("slogans", [])
            ),
            "logo_success_count": len(
                brand_result.get("logos", [])
            ),
        }
        print("  metadata 없음 → 자동 추가")
        changed = True

    # 최신 구조라도 누락된 metadata 하위 키는 보완
    defaults = {
        "status": "success",
        "saved_at": "",
        "output_directory": portable_output_directory(output_dir),
        "brand_name_count": len(
            brand_result.get("brand_names", [])
        ),
        "slogan_count": len(
            brand_result.get("slogans", [])
        ),
        "logo_success_count": len(
            brand_result.get("logos", [])
        ),
    }

    for key, value in defaults.items():
        if key not in brand_result["metadata"]:
            brand_result["metadata"][key] = value
            changed = True

    return changed


# =========================================================
# 저장 / 불러오기
# =========================================================
def save_brand_result(
    brand_result,
    output_dir,
    show_summary=True,
):
    """brand_result.json을 저장하고 metadata를 갱신합니다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir / "brand_result.json"
    )

    finalize_metadata(
        brand_result,
        output_dir,
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            brand_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    if show_summary:
        print_summary(
            brand_result
        )

    return result_path


def load_brand_result(result_path):
    """
    기존 brand_result.json을 불러오고,
    구버전 구조면 자동 보완 후 재저장합니다.
    """

    result_path = Path(result_path)

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as file:
        brand_result = json.load(
            file
        )

    if not isinstance(
        brand_result,
        dict,
    ):
        raise TypeError(
            "brand_result.json의 최상위 형식은 "
            "JSON 객체(dict)여야 합니다."
        )

    changed = migrate_brand_result(
        brand_result,
        result_path,
    )

    if changed:
        save_brand_result(
            brand_result,
            result_path.parent,
            show_summary=False,
        )
        print(
            "  기존 brand_result.json 구조 보완 완료"
        )
    else:
        print(
            "  기존 brand_result.json은 최신 구조입니다."
        )

    return brand_result


# =========================================================
# 공통 API 재시도
# =========================================================
def call_with_retry(
    api_func,
    item_name,
    brand_result,
    stage,
):
    """
    API 호출을 최대 MAX_RETRIES회 재시도합니다.
    최종 실패한 경우에만 errors 배열에 기록합니다.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            result = api_func()

            if attempt > 1:
                print(
                    f"  ✅ {item_name} — 성공 "
                    f"(시도 {attempt}회)"
                )

            remove_error(
                brand_result,
                stage,
                item_name,
            )

            return result

        except Exception as error:
            last_error = error

            print(
                f"  ⚠️ {item_name} — 실패 "
                f"(시도 {attempt}/{MAX_RETRIES}): "
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[
                    attempt - 1
                ]

                print(
                    f"     {delay}초 후 재시도합니다..."
                )

                time.sleep(
                    delay
                )

    add_error(
        brand_result,
        stage=stage,
        item=item_name,
        message=str(last_error),
        attempts=MAX_RETRIES,
    )

    print(
        f"  ❌ {item_name} — 최종 실패"
    )

    return None


# =========================================================
# LLM 프롬프트 생성
# =========================================================
def build_brand_prompt(brief):
    """브랜드 생성용 최초 프롬프트를 만듭니다."""

    brief_json = json.dumps(
        brief,
        ensure_ascii=False,
        indent=2,
    )

    style_block = ""

    if brief.get(
        "style_examples"
    ):
        examples_str = "\n".join(
            f"- {example}"
            for example
            in brief["style_examples"]
        )

        style_block = f"""
[문체 참고 예시]
{examples_str}

위 문장을 그대로 복사하지 말고,
문장의 길이·톤·전문적이고 간결한 분위기만 참고하세요.
"""

    return f"""
당신은 전문 브랜드 전략가입니다.

아래 제품 브랜드 브리프를 분석하여
새로운 브랜드 아이덴티티를 만들어주세요.

[브랜드 브리프]
{brief_json}
{style_block}
다음 조건을 반드시 지켜주세요.

1. 브랜드명 후보를 3~5개 생성합니다.
2. 각 브랜드명에는 name과 meaning을 작성합니다.
3. 타깃과 브랜드 톤에 맞는 슬로건을 정확히 3개 생성합니다.
4. 브랜드 스토리는 약 300자 분량의 한국어로 작성합니다.
5. 브랜드 스토리에는 브랜드 탄생 배경, 브랜드 철학, 브랜드 비전을 포함합니다.
6. 컬러 팔레트는 메인 컬러 1개와 서브 컬러 2~3개로 구성합니다.
7. 모든 컬러는 name과 hex를 작성합니다.
8. HEX 코드는 반드시 #RRGGBB 형식을 사용합니다.
9. 제품 특징, 타깃, 핵심 키워드, 브랜드 톤을 충분히 반영합니다.

반드시 아래 JSON 형식으로만 답변하세요.

{{
  "brand_names": [
    {{
      "name": "브랜드명",
      "meaning": "브랜드명의 의미"
    }}
  ],
  "slogans": [
    "슬로건 1",
    "슬로건 2",
    "슬로건 3"
  ],
  "brand_story": "약 300자의 브랜드 스토리",
  "colors": {{
    "main": {{
      "name": "메인 컬러 이름",
      "hex": "#000000"
    }},
    "sub": [
      {{
        "name": "서브 컬러 이름",
        "hex": "#000000"
      }}
    ]
  }}
}}
""".strip()


# =========================================================
# LLM 응답 JSON 추출 / 검증
# =========================================================
def strip_code_fence(text):
    """```json ... ``` 형태의 코드펜스를 제거합니다."""

    text = text.strip()

    if text.startswith("```"):
        if text.startswith("```json"):
            text = text[len("```json"):]
        else:
            text = text[len("```"):]

        if text.endswith("```"):
            text = text[:-3]

    return text.strip()


def validate_llm_result(data):
    """
    실제 과제의 brand_result 스키마를 검사합니다.
    반환값이 빈 리스트이면 검증 통과입니다.
    """

    errors = []

    if not isinstance(
        data,
        dict,
    ):
        return [
            "LLM 결과의 최상위 형식은 JSON 객체여야 합니다."
        ]

    # brand_names: {name, meaning} 객체 3~5개
    names = data.get(
        "brand_names",
        [],
    )

    if not isinstance(
        names,
        list,
    ):
        errors.append(
            "brand_names는 리스트여야 합니다."
        )

    elif not (
        3 <= len(names) <= 5
    ):
        errors.append(
            "brand_names는 "
            f"{len(names)}개입니다. "
            "3~5개여야 합니다."
        )

    else:
        for index, item in enumerate(
            names,
            start=1,
        ):
            if not isinstance(
                item,
                dict,
            ):
                errors.append(
                    f"brand_names[{index}]는 객체여야 합니다."
                )
                continue

            for field in [
                "name",
                "meaning",
            ]:
                value = item.get(
                    field
                )

                if (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value.strip()
                ):
                    errors.append(
                        f"brand_names[{index}].{field}는 "
                        "비어 있지 않은 문자열이어야 합니다."
                    )

    # slogans: 정확히 3개
    slogans = data.get(
        "slogans",
        [],
    )

    if not isinstance(
        slogans,
        list,
    ):
        errors.append(
            "slogans는 리스트여야 합니다."
        )

    elif len(
        slogans
    ) != 3:
        errors.append(
            f"slogans는 {len(slogans)}개입니다. "
            "정확히 3개여야 합니다."
        )

    elif not all(
        isinstance(
            slogan,
            str,
        )
        and slogan.strip()
        for slogan in slogans
    ):
        errors.append(
            "slogans의 모든 항목은 "
            "비어 있지 않은 문자열이어야 합니다."
        )

    # brand_story
    story = data.get(
        "brand_story",
        "",
    )

    if (
        not isinstance(
            story,
            str,
        )
        or not story.strip()
    ):
        errors.append(
            "brand_story는 비어 있지 않은 문자열이어야 합니다."
        )

    # colors.main / colors.sub
    colors = data.get(
        "colors",
        {},
    )

    if not isinstance(
        colors,
        dict,
    ):
        errors.append(
            "colors는 객체여야 합니다."
        )
        return errors

    main_color = colors.get(
        "main"
    )

    if not isinstance(
        main_color,
        dict,
    ):
        errors.append(
            "colors.main은 객체여야 합니다."
        )

    else:
        if (
            not isinstance(
                main_color.get("name"),
                str,
            )
            or not main_color.get(
                "name",
                "",
            ).strip()
        ):
            errors.append(
                "colors.main.name은 "
                "비어 있지 않은 문자열이어야 합니다."
            )

        if not validate_hex_color(
            main_color.get(
                "hex",
                "",
            )
        ):
            errors.append(
                "colors.main.hex는 "
                "#RRGGBB 형식이어야 합니다."
            )

    sub_colors = colors.get(
        "sub",
        [],
    )

    if not isinstance(
        sub_colors,
        list,
    ):
        errors.append(
            "colors.sub는 리스트여야 합니다."
        )

    elif not (
        2 <= len(
            sub_colors
        ) <= 3
    ):
        errors.append(
            f"colors.sub는 {len(sub_colors)}개입니다. "
            "2~3개여야 합니다."
        )

    else:
        for index, item in enumerate(
            sub_colors,
            start=1,
        ):
            if not isinstance(
                item,
                dict,
            ):
                errors.append(
                    f"colors.sub[{index}]는 객체여야 합니다."
                )
                continue

            if (
                not isinstance(
                    item.get("name"),
                    str,
                )
                or not item.get(
                    "name",
                    "",
                ).strip()
            ):
                errors.append(
                    f"colors.sub[{index}].name은 "
                    "비어 있지 않은 문자열이어야 합니다."
                )

            if not validate_hex_color(
                item.get(
                    "hex",
                    "",
                )
            ):
                errors.append(
                    f"colors.sub[{index}].hex는 "
                    "#RRGGBB 형식이어야 합니다."
                )

    return errors


SCHEMA_DESCRIPTION = """- brand_names: 객체 리스트 3~5개, 각 항목에 name과 meaning 포함
- slogans: 비어 있지 않은 문자열 리스트, 정확히 3개
- brand_story: 비어 있지 않은 문자열(약 300자)
- colors.main: name과 hex를 가진 객체
- colors.sub: name과 hex를 가진 객체 리스트 2~3개
- 모든 hex: #RRGGBB 형식"""


def build_retry_prompt(
    original_prompt,
    raw_response,
    error_messages,
):
    """스키마 오류를 LLM이 스스로 수정하도록 재질문 프롬프트를 만듭니다."""

    error_list = "\n".join(
        f"- {message}"
        for message in error_messages
    )

    return f"""
아래는 원래 브랜드 생성 요청입니다.

[원래 요청]
{original_prompt}

이전 응답이 요구 스키마와 맞지 않았습니다.
발견된 오류를 모두 수정하여 JSON만 다시 출력하세요.

[발견된 오류]
{error_list}

[이전 응답]
{raw_response}

[요구 스키마]
{SCHEMA_DESCRIPTION}

설명문, Markdown 코드펜스 없이
수정된 JSON 객체만 출력하세요.
""".strip()


# =========================================================
# LLM 호출 + 검증 + 자동 재질문
# =========================================================
def request_llm_with_validation(
    client,
    brief,
    brand_result,
):
    """
    LLM 결과를 파싱/검증하고,
    실패하면 오류 내용을 반영해 최대 3회 수정 요청합니다.
    """

    original_prompt = build_brand_prompt(
        brief
    )

    prompt = original_prompt
    last_errors = []

    for validation_attempt in range(
        1,
        LLM_VALIDATION_RETRIES + 1,
    ):
        response = call_with_retry(
            api_func=lambda: client.responses.create(
                model="gpt-5.6-luna",
                input=prompt,
            ),
            item_name=(
                f"brand_text_api_{validation_attempt}"
            ),
            brand_result=brand_result,
            stage="llm_api",
        )

        if response is None:
            # 네트워크/API 최종 실패는 call_with_retry가 기록
            return None

        raw_response = response.output_text.strip()
        result_text = strip_code_fence(
            raw_response
        )

        try:
            data = json.loads(
                result_text
            )

        except json.JSONDecodeError as error:
            last_errors = [
                "JSON 파싱 실패: "
                f"{error.msg}"
            ]

        else:
            last_errors = validate_llm_result(
                data
            )

            if not last_errors:
                remove_error(
                    brand_result,
                    "llm_validation",
                    "brand_result",
                )

                print(
                    "  ✅ LLM 응답 검증 통과 "
                    f"(시도 {validation_attempt}회)"
                )

                return data

        print(
            "  ⚠️ LLM 응답 검증 실패 "
            f"(시도 {validation_attempt}/"
            f"{LLM_VALIDATION_RETRIES})"
        )

        for message in last_errors:
            print(
                f"     - {message}"
            )

        if (
            validation_attempt
            < LLM_VALIDATION_RETRIES
        ):
            prompt = build_retry_prompt(
                original_prompt,
                raw_response,
                last_errors,
            )

    add_error(
        brand_result,
        stage="llm_validation",
        item="brand_result",
        message=" | ".join(
            last_errors
        ),
        attempts=LLM_VALIDATION_RETRIES,
    )

    print(
        "  ❌ LLM 응답 검증 최종 실패"
    )

    return None


def generate_brand_identity(
    client,
    brief,
    brand_result=None,
):
    """
    기존 호출 호환성을 유지하면서
    검증/자동 재질문이 포함된 브랜드 결과를 반환합니다.
    """

    if brand_result is None:
        brand_result = build_brand_result()

    result = request_llm_with_validation(
        client,
        brief,
        brand_result,
    )

    if result is None:
        raise ValueError(
            "브랜드 아이덴티티 생성에 최종 실패했습니다."
        )

    return result


# =========================================================
# HEX 컬러 코드 검사 / 대체
# =========================================================
def validate_hex_color(hex_code):
    if not isinstance(
        hex_code,
        str,
    ):
        return False

    return bool(
        HEX_PATTERN.fullmatch(
            hex_code
        )
    )


def sanitize_hex(
    value,
    field_name,
    brand_result,
    default_hex,
):
    """
    HEX 형식이 잘못된 기존 결과를 기본값으로 대체하고
    errors에 기록합니다.
    """

    if validate_hex_color(
        value
    ):
        return value.upper()

    add_error(
        brand_result,
        stage="color_validation",
        item=field_name,
        message=(
            f"HEX 형식 오류({value!r}) → "
            f"기본값 {default_hex}로 대체"
        ),
        attempts=1,
    )

    return default_hex


def sanitize_result_colors(
    brand_result
):
    """기존/마이그레이션 결과의 HEX를 안전하게 정리합니다."""

    colors = brand_result.setdefault(
        "colors",
        {}
    )

    main_color = colors.setdefault(
        "main",
        {
            "name": "프로페셔널 딥 네이비",
            "hex": DEFAULT_MAIN_HEX,
        },
    )

    main_color["hex"] = sanitize_hex(
        main_color.get(
            "hex",
            "",
        ),
        "colors.main.hex",
        brand_result,
        DEFAULT_MAIN_HEX,
    )

    sub_colors = colors.get(
        "sub",
        [],
    )

    if not isinstance(
        sub_colors,
        list,
    ):
        sub_colors = []

    # 부족한 서브 컬러는 최소 2개가 되도록 기본값 보완
    while len(
        sub_colors
    ) < 2:
        default_index = len(
            sub_colors
        )

        sub_colors.append(
            {
                "name": (
                    f"기본 서브 컬러 "
                    f"{default_index + 1}"
                ),
                "hex": DEFAULT_SUB_HEXES[
                    default_index
                ],
            }
        )

        add_error(
            brand_result,
            stage="color_validation",
            item="colors.sub",
            message=(
                "서브 컬러가 2개 미만이어서 "
                "기본 색상으로 보완했습니다."
            ),
            attempts=1,
        )

    # 과제 요구 최대 3개
    sub_colors = sub_colors[:3]

    for index, sub_color in enumerate(
        sub_colors
    ):
        if not isinstance(
            sub_color,
            dict,
        ):
            sub_color = {
                "name": (
                    f"기본 서브 컬러 "
                    f"{index + 1}"
                ),
                "hex": DEFAULT_SUB_HEXES[
                    index
                ],
            }
            sub_colors[index] = sub_color

        sub_color.setdefault(
            "name",
            f"서브 컬러 {index + 1}",
        )

        sub_color["hex"] = sanitize_hex(
            sub_color.get(
                "hex",
                "",
            ),
            f"colors.sub[{index}].hex",
            brand_result,
            DEFAULT_SUB_HEXES[
                index
            ],
        )

    colors["sub"] = sub_colors


# =========================================================
# 컬러 팔레트 PNG 생성
# =========================================================
def generate_color_palette(
    brand_result,
    output_dir,
):
    """
    brand_result의 colors 정보를 이용해
    가로형 컬러 팔레트 PNG를 생성합니다.
    """

    sanitize_result_colors(
        brand_result
    )

    colors = brand_result.get(
        "colors",
        {}
    )

    main_color = colors.get(
        "main",
        {},
    )

    sub_colors = colors.get(
        "sub",
        [],
    )

    palette_colors = [
        {
            "type": "MAIN COLOR",
            "name": main_color.get(
                "name",
                "Main Color",
            ),
            "hex": main_color.get(
                "hex",
                DEFAULT_MAIN_HEX,
            ),
        }
    ]

    for sub_color in sub_colors:
        palette_colors.append(
            {
                "type": "SUB COLOR",
                "name": sub_color.get(
                    "name",
                    "Sub Color",
                ),
                "hex": sub_color.get(
                    "hex",
                    DEFAULT_SUB_HEXES[0],
                ),
            }
        )

    plt.rcParams[
        "font.family"
    ] = "Malgun Gothic"

    plt.rcParams[
        "axes.unicode_minus"
    ] = False

    color_count = len(
        palette_colors
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    ax.set_xlim(
        0,
        color_count,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis(
        "off"
    )

    fig.suptitle(
        "Brand Color Palette",
        fontsize=22,
        fontweight="bold",
        y=0.94,
    )

    for index, color in enumerate(
        palette_colors
    ):
        rectangle = Rectangle(
            (
                index,
                0.35,
            ),
            1,
            0.45,
            facecolor=color["hex"],
            edgecolor="none",
        )

        ax.add_patch(
            rectangle
        )

        ax.text(
            index + 0.5,
            0.27,
            color["type"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            index + 0.5,
            0.18,
            color["name"],
            ha="center",
            va="center",
            fontsize=11,
        )

        ax.text(
            index + 0.5,
            0.09,
            color["hex"].upper(),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    palette_path = (
        Path(output_dir)
        / "brand_palette.png"
    )

    plt.savefig(
        palette_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    return palette_path


# =========================================================
# 로고 생성 프롬프트 만들기
# =========================================================
def build_logo_prompts(
    brand_result,
    brief,
):
    """대표 브랜드에 맞는 서로 다른 로고 콘셉트 3개를 만듭니다."""

    brand_names = brand_result.get(
        "brand_names",
        [],
    )

    if not brand_names:
        raise ValueError(
            "생성된 브랜드명이 없습니다."
        )

    brand_name = brand_names[0].get(
        "name",
        "Brand",
    )

    tone = brief.get(
        "tone",
        "세련되고 전문적인",
    )

    target = brief.get(
        "target",
        "",
    )

    keywords = ", ".join(
        brief.get(
            "keywords",
            [],
        )
    )

    colors = brand_result.get(
        "colors",
        {},
    )

    main_hex = colors.get(
        "main",
        {},
    ).get(
        "hex",
        DEFAULT_MAIN_HEX,
    )

    sub_hexes = [
        item.get(
            "hex",
            "",
        )
        for item in colors.get(
            "sub",
            [],
        )
    ]

    color_text = ", ".join(
        [
            main_hex,
            *sub_hexes,
        ]
    )

    common = f"""
Create a professional brand logo for "{brand_name}".

Product category:
premium oral irrigator / oral-care appliance.

Primary target:
{target}

Brand keywords:
{keywords}

Brand tone:
{tone}

Brand colors:
{color_text}

Design requirements:
minimal, modern, clean, premium, professional,
simple geometric vector-like logo,
white background,
high visual clarity,
balanced spacing,
suitable for product packaging and web branding,
no mockup,
no photograph,
no 3D render,
no extra decorative background.
"""

    prompt_01 = common + """
Concept 1:
Combine a clean water droplet symbol
with a subtle tooth shape.
The icon should communicate precision water cleaning and oral hygiene.
Place the brand name clearly below or beside the symbol.
"""

    prompt_02 = common + """
Concept 2:
Create a flowing water-wave symbol
combined with a clean modern wordmark.
The visual should suggest controlled water pressure,
daily oral-care routine, efficiency and freshness.
"""

    prompt_03 = common + """
Concept 3:
Create a distinctive monogram-style symbol
using the initial letter of the brand name.
Integrate a subtle dental or water motif.
Keep the result highly minimal,
professional and recognizable at small sizes.
"""

    return [
        prompt_01,
        prompt_02,
        prompt_03,
    ]


# =========================================================
# 이미지 API로 로고 생성
# =========================================================
def generate_logo_images(
    client,
    brand_result,
    brief,
    output_dir,
):
    """
    GPT-Image-2로 로고 3개를 생성합니다.
    각 로고는 최대 3회 재시도하고 최종 실패만 errors에 기록합니다.
    """

    prompts = build_logo_prompts(
        brand_result,
        brief,
    )

    output_dir = Path(
        output_dir
    )

    generated_paths = []
    brand_result.setdefault(
        "logos",
        [],
    )

    for index, prompt in enumerate(
        prompts,
        start=1,
    ):
        item_name = (
            f"logo_{index:02d}"
        )

        logo_filename = (
            f"{item_name}.png"
        )

        logo_path = (
            output_dir
            / logo_filename
        )

        print(
            f"\n[{item_name}] 생성 시작"
        )

        def _request_image():
            response = client.images.generate(
                model="gpt-image-2",
                prompt=prompt,
                size="1024x1024",
                quality="low",
            )

            if (
                not response.data
                or not response.data[0].b64_json
            ):
                raise ValueError(
                    "이미지 데이터가 반환되지 않았습니다."
                )

            return response.data[
                0
            ].b64_json

        image_base64 = call_with_retry(
            api_func=_request_image,
            item_name=item_name,
            brand_result=brand_result,
            stage="logo_generation",
        )

        if image_base64 is None:
            print(
                "  다음 로고 생성을 계속합니다."
            )
            continue

        image_bytes = base64.b64decode(
            image_base64
        )

        with open(
            logo_path,
            "wb",
        ) as image_file:
            image_file.write(
                image_bytes
            )

        generated_paths.append(
            logo_path
        )

        # 상대 파일명만 결과 JSON에 저장
        if logo_filename not in brand_result[
            "logos"
        ]:
            brand_result[
                "logos"
            ].append(
                logo_filename
            )

        print(
            f"  ✅ 저장 완료: {logo_path}"
        )

    return generated_paths


# =========================================================
# 브랜드 브리프 유효성 검사
# =========================================================
def validate_brief(brief):
    if not isinstance(
        brief,
        dict,
    ):
        raise TypeError(
            "브리프 데이터의 최상위 형식은 "
            "JSON 객체(dict)여야 합니다."
        )

    required_fields = [
        "industry",
        "target",
        "keywords",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in brief
    ]

    if missing_fields:
        raise ValueError(
            "필수 항목이 없습니다: "
            + ", ".join(
                missing_fields
            )
        )

    for field in [
        "industry",
        "target",
    ]:
        if (
            not isinstance(
                brief[field],
                str,
            )
            or not brief[
                field
            ].strip()
        ):
            raise ValueError(
                f"{field}는 비어 있지 않은 "
                "문자열이어야 합니다."
            )

    if (
        not isinstance(
            brief["keywords"],
            list,
        )
        or not brief[
            "keywords"
        ]
    ):
        raise ValueError(
            "keywords는 비어 있지 않은 "
            "리스트여야 합니다."
        )

    if not all(
        isinstance(
            keyword,
            str,
        )
        and keyword.strip()
        for keyword in brief[
            "keywords"
        ]
    ):
        raise ValueError(
            "keywords의 각 항목은 "
            "비어 있지 않은 문자열이어야 합니다."
        )

    if "product_name" in brief:
        if (
            not isinstance(
                brief["product_name"],
                str,
            )
            or not brief[
                "product_name"
            ].strip()
        ):
            raise ValueError(
                "product_name은 비어 있지 않은 "
                "문자열이어야 합니다."
            )

    if "tone" in brief:
        if (
            not isinstance(
                brief["tone"],
                str,
            )
            or not brief[
                "tone"
            ].strip()
        ):
            raise ValueError(
                "tone은 비어 있지 않은 "
                "문자열이어야 합니다."
            )

    if "competitors" in brief:
        if (
            not isinstance(
                brief["competitors"],
                list,
            )
            or not brief[
                "competitors"
            ]
        ):
            raise ValueError(
                "competitors는 비어 있지 않은 "
                "리스트여야 합니다."
            )

        if not all(
            isinstance(
                competitor,
                str,
            )
            and competitor.strip()
            for competitor in brief[
                "competitors"
            ]
        ):
            raise ValueError(
                "competitors의 각 항목은 "
                "비어 있지 않은 문자열이어야 합니다."
            )

    if "notes" in brief:
        if (
            not isinstance(
                brief["notes"],
                str,
            )
            or not brief[
                "notes"
            ].strip()
        ):
            raise ValueError(
                "notes는 비어 있지 않은 "
                "문자열이어야 합니다."
            )

    # style_examples는 선택 항목이며 기존 브리프와 하위 호환
    if "style_examples" in brief:
        examples = brief[
            "style_examples"
        ]

        if not isinstance(
            examples,
            list,
        ):
            raise ValueError(
                "style_examples는 리스트여야 합니다."
            )

        if not examples:
            raise ValueError(
                "style_examples는 입력하는 경우 "
                "1개 이상의 문장을 포함해야 합니다."
            )

        if not all(
            isinstance(
                example,
                str,
            )
            and example.strip()
            for example in examples
        ):
            raise ValueError(
                "style_examples의 모든 항목은 "
                "비어 있지 않은 문자열이어야 합니다."
            )


# =========================================================
# 브랜드 브리프 불러오기
# =========================================================
def load_brand_brief(file_path):
    if not file_path:
        raise ValueError(
            "파일 경로를 입력해주세요."
        )

    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            "파일을 찾을 수 없습니다: "
            f"{file_path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        brief = json.load(
            file
        )

    validate_brief(
        brief
    )

    return brief


# =========================================================
# 메인 프로그램
# =========================================================
def main():
    print(
        "=" * 40
    )
    print(
        " AI 브랜드 아이덴티티 생성기"
    )
    print(
        "=" * 40
    )

    while True:
        brief_path = input(
            "\n브랜드 브리프 JSON 파일 경로를 입력하세요.\n"
            "예) briefs/wp670k_brief.json\n"
            "※ 상대경로와 절대경로 모두 사용할 수 있습니다.\n"
            "> "
        ).strip()

        if brief_path:
            break

        print(
            "[오류] 파일 경로는 필수 입력입니다."
        )

    output_path = input(
        "\n출력 폴더 경로를 입력하세요.\n"
        "예) output/wp670k\n"
        "(Enter 입력 시 ./output 사용)\n"
        "> "
    ).strip()

    if not output_path:
        output_path = "./output"

    try:
        # ① 입력 검증
        brief = load_brand_brief(
            brief_path
        )

        output_dir = Path(
            output_path
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        product_name = brief.get(
            "product_name",
            "제품명 없음",
        )

        print(
            "\n브랜드 브리프를 불러왔습니다."
        )
        print(
            f"제품: {product_name}"
        )
        print(
            f"산업 분야: {brief['industry']}"
        )
        print(
            f"타깃: {brief['target']}"
        )
        print(
            "핵심 키워드: "
            + ", ".join(
                brief["keywords"]
            )
        )
        print(
            f"출력 폴더: {output_path}"
        )

        result_path = (
            output_dir
            / "brand_result.json"
        )

        # ② 기존 JSON 확인 → API 재호출 없이 구조 보완
        if result_path.exists():
            print(
                "\n기존 brand_result.json을 찾았습니다."
            )
            print(
                "기존 결과를 사용하므로 "
                "텍스트 생성 API를 다시 호출하지 않습니다."
            )

            brand_result = load_brand_result(
                result_path
            )

        # ③ 신규 생성 → LLM 검증 + 자동 수정 재질문
        else:
            print(
                "\n신규 브랜드 아이덴티티 생성을 시작합니다..."
            )

            brand_result = build_brand_result()
            client = get_openai_client()

            data = request_llm_with_validation(
                client,
                brief,
                brand_result,
            )

            if data is None:
                print(
                    "\n텍스트 생성 최종 실패 → "
                    "오류 메타를 저장하고 실행을 중단합니다."
                )

                save_brand_result(
                    brand_result,
                    output_dir,
                )
                return

            brand_result[
                "brand_names"
            ] = data[
                "brand_names"
            ]

            brand_result[
                "slogans"
            ] = data[
                "slogans"
            ]

            brand_result[
                "brand_story"
            ] = data[
                "brand_story"
            ]

            brand_result[
                "colors"
            ] = data[
                "colors"
            ]

            save_brand_result(
                brand_result,
                output_dir,
                show_summary=False,
            )

            print(
                "브랜드 아이덴티티 생성 및 저장 완료"
            )

        # ④ 컬러 생성
        print(
            "\n컬러 팔레트 이미지를 생성합니다..."
        )

        palette_path = generate_color_palette(
            brand_result,
            output_dir,
        )

        print(
            "컬러 팔레트 이미지 생성 완료:"
        )
        print(
            palette_path
        )

        # HEX 대체 오류 등이 있으면 즉시 저장
        save_brand_result(
            brand_result,
            output_dir,
            show_summary=False,
        )

        # ⑤ 로고 생성: 과제 요구에 맞춰 정확히 3개 시안
        print(
            "\n로고 이미지 3개를 이미지 API로 "
            "생성할 수 있습니다."
        )
        print(
            "※ 이미지 생성 API 비용이 발생합니다."
        )

        create_logos = input(
            "지금 로고 이미지를 생성하시겠습니까? "
            "(y/n): "
        ).strip().lower()

        if create_logos == "y":
            client = get_openai_client()

            print(
                "\n로고 이미지 생성을 시작합니다..."
            )

            generate_logo_images(
                client,
                brand_result,
                brief,
                output_dir,
            )

        else:
            print(
                "\n로고 이미지 생성을 건너뜁니다."
            )

        # ⑥ 최종 저장 + 요약
        result_path = save_brand_result(
            brand_result,
            output_dir,
            show_summary=True,
        )

        print(
            "\n최종 결과 저장 완료:"
        )
        print(
            result_path
        )
        print(
            "\n프로그램 실행이 완료되었습니다."
        )

    except FileNotFoundError as error:
        print(
            f"\n[파일 오류] {error}"
        )

    except json.JSONDecodeError:
        print(
            "\n[JSON 오류] "
            "JSON 형식을 확인해주세요."
        )

    except PermissionError:
        print(
            "\n[폴더 오류] "
            "출력 폴더에 파일을 저장할 권한이 없습니다."
        )

    except TypeError as error:
        print(
            f"\n[형식 오류] {error}"
        )

    except ValueError as error:
        print(
            f"\n[입력 오류] {error}"
        )

    except OSError as error:
        print(
            f"\n[시스템 오류] {error}"
        )

    except Exception as error:
        print(
            f"\n[프로그램 오류] {error}"
        )


if __name__ == "__main__":
    main()
