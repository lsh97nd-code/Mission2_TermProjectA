import base64
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# OpenAI 설정
# =========================================================
def get_openai_client():
    """
    .env 파일에서 OpenAI API 키를 불러와
    OpenAI 클라이언트를 생성합니다.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다."
        )

    return OpenAI(api_key=api_key)


# =========================================================
# 브랜드 아이덴티티 생성
# =========================================================
def generate_brand_identity(client, brief):
    """
    브랜드 브리프를 기반으로
    브랜드명, 슬로건, 브랜드 스토리,
    컬러 정보를 생성합니다.
    """

    brief_json = json.dumps(
        brief,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
당신은 전문 브랜드 전략가입니다.

아래 제품 브랜드 브리프를 분석하여
새로운 브랜드 아이덴티티를 만들어주세요.

[브랜드 브리프]
{brief_json}

다음 조건을 반드시 지켜주세요.

1. 브랜드명 후보를 3~5개 생성합니다.
2. 각 브랜드명에는 이름과 의미를 작성합니다.
3. 타깃과 브랜드 톤에 맞는 슬로건을 정확히 3개 생성합니다.
4. 브랜드 스토리는 약 300자 분량의 한국어로 작성합니다.
5. 브랜드 스토리에는 브랜드 탄생 배경,
   브랜드 철학, 브랜드 비전을 포함합니다.
6. 컬러 팔레트를 생성합니다.
   - 메인 컬러 1개
   - 서브 컬러 2~3개
7. 각 컬러에는 색상 이름과 HEX 코드를 작성합니다.
8. HEX 코드는 반드시 #RRGGBB 형식을 사용합니다.
9. 제품 특징, 타깃, 핵심 키워드,
   브랜드 톤을 충분히 반영합니다.

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
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    result_text = response.output_text.strip()

    if result_text.startswith("```"):

        if result_text.startswith("```json"):
            result_text = result_text[len("```json"):]
        else:
            result_text = result_text[len("```"):]

        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result_text = result_text.strip()

    try:
        result = json.loads(result_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "OpenAI 응답을 JSON으로 변환하지 못했습니다."
        ) from error

    return result


# =========================================================
# 브랜드 결과 저장
# =========================================================
def save_brand_result(result, output_dir):
    """
    브랜드 결과를 brand_result.json으로 저장합니다.
    """

    result_path = output_dir / "brand_result.json"

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    return result_path


# =========================================================
# 기존 브랜드 결과 읽기
# =========================================================
def load_brand_result(result_path):
    """
    기존 brand_result.json을 불러옵니다.
    """

    with open(
        result_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# HEX 컬러 코드 검사
# =========================================================
def validate_hex_color(hex_code):

    if not isinstance(hex_code, str):
        return False

    pattern = r"^#[0-9A-Fa-f]{6}$"

    return bool(
        re.match(
            pattern,
            hex_code
        )
    )


# =========================================================
# 컬러 팔레트 PNG 생성
# =========================================================
def generate_color_palette(
    result_path,
    output_dir
):
    """
    brand_result.json의 colors 정보를 읽어서
    컬러 팔레트 PNG 이미지를 생성합니다.
    """

    brand_result = load_brand_result(
        result_path
    )

    colors = brand_result.get(
        "colors"
    )

    if not colors:
        raise ValueError(
            "brand_result.json에 colors 정보가 없습니다."
        )

    main_color = colors.get(
        "main"
    )

    sub_colors = colors.get(
        "sub",
        []
    )

    if not main_color:
        raise ValueError(
            "메인 컬러 정보가 없습니다."
        )

    palette_colors = [
        {
            "type": "MAIN COLOR",
            "name": main_color.get(
                "name",
                "Main Color"
            ),
            "hex": main_color.get(
                "hex",
                ""
            )
        }
    ]

    for sub_color in sub_colors:

        palette_colors.append(
            {
                "type": "SUB COLOR",
                "name": sub_color.get(
                    "name",
                    "Sub Color"
                ),
                "hex": sub_color.get(
                    "hex",
                    ""
                )
            }
        )

    for color in palette_colors:

        if not validate_hex_color(
            color["hex"]
        ):
            raise ValueError(
                "잘못된 HEX 컬러 코드입니다: "
                f"{color['hex']}"
            )

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    color_count = len(
        palette_colors
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            6
        )
    )

    ax.set_xlim(
        0,
        color_count
    )

    ax.set_ylim(
        0,
        1
    )

    ax.axis(
        "off"
    )

    fig.suptitle(
        "Brand Color Palette",
        fontsize=22,
        fontweight="bold",
        y=0.94
    )

    for index, color in enumerate(
        palette_colors
    ):

        rectangle = Rectangle(
            (
                index,
                0.35
            ),
            1,
            0.45,
            facecolor=color["hex"],
            edgecolor="none"
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
            fontweight="bold"
        )

        ax.text(
            index + 0.5,
            0.18,
            color["name"],
            ha="center",
            va="center",
            fontsize=11
        )

        ax.text(
            index + 0.5,
            0.09,
            color["hex"].upper(),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

    palette_path = (
        output_dir
        / "brand_palette.png"
    )

    plt.savefig(
        palette_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
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
    brief
):
    """
    브랜드 정보에 맞는
    서로 다른 로고 콘셉트 3개를 생성합니다.
    """

    brand_names = brand_result.get(
        "brand_names",
        []
    )

    if not brand_names:
        raise ValueError(
            "생성된 브랜드명이 없습니다."
        )

    # 첫 번째 브랜드명을 대표 브랜드명으로 사용
    brand_name = brand_names[0].get(
        "name",
        "Brand"
    )

    tone = brief.get(
        "tone",
        "세련되고 전문적인"
    )

    target = brief.get(
        "target",
        ""
    )

    keywords = ", ".join(
        brief.get(
            "keywords",
            []
        )
    )

    colors = brand_result.get(
        "colors",
        {}
    )

    main_hex = colors.get(
        "main",
        {}
    ).get(
        "hex",
        "#12304A"
    )

    sub_hexes = [
        item.get(
            "hex",
            ""
        )
        for item in colors.get(
            "sub",
            []
        )
    ]

    color_text = ", ".join(
        [
            main_hex
        ] + sub_hexes
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
The icon should communicate
precision water cleaning and oral hygiene.
Place the brand name clearly below or beside the symbol.
"""

    prompt_02 = common + """
Concept 2:
Create a flowing water-wave symbol
combined with a clean modern wordmark.
The visual should suggest
controlled water pressure,
daily oral-care routine,
efficiency and freshness.
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
        prompt_03
    ]


# =========================================================
# 이미지 API로 로고 생성
# =========================================================
def generate_logo_images(
    client,
    brand_result,
    brief,
    output_dir
):
    """
    GPT-Image-2 이미지 API로
    로고 3개를 생성해 PNG 파일로 저장합니다.

    각 로고 생성 실패 시 오류를 출력하고
    다음 로고 생성을 계속 진행합니다.
    """

    prompts = build_logo_prompts(
        brand_result,
        brief
    )

    generated_paths = []

    for index, prompt in enumerate(
        prompts,
        start=1
    ):

        logo_filename = (
            f"logo_{index:02d}.png"
        )

        logo_path = (
            output_dir
            / logo_filename
        )

        print(
            f"\n로고 {index}/3 생성 중..."
        )

        try:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=prompt,
                size="1024x1024",
                quality="low"
            )

            if (
                not response.data
                or not response.data[0].b64_json
            ):
                raise ValueError(
                    "이미지 데이터가 반환되지 않았습니다."
                )

            image_base64 = (
                response.data[0].b64_json
            )

            image_bytes = (
                base64.b64decode(
                    image_base64
                )
            )

            with open(
                logo_path,
                "wb"
            ) as image_file:

                image_file.write(
                    image_bytes
                )

            generated_paths.append(
                logo_path
            )

            print(
                f"로고 {index}/3 생성 완료: "
                f"{logo_path}"
            )

        except Exception as error:

            print(
                f"[로고 {index} 오류] "
                f"{error}"
            )

            print(
                "다음 로고 생성을 계속합니다."
            )

    return generated_paths


# =========================================================
# 브랜드 브리프 유효성 검사
# =========================================================
def validate_brief(brief):

    if not isinstance(
        brief,
        dict
    ):
        raise TypeError(
            "브리프 데이터의 최상위 형식은 "
            "JSON 객체(dict)여야 합니다."
        )

    required_fields = [
        "industry",
        "target",
        "keywords"
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
        "target"
    ]:

        if (
            not isinstance(
                brief[field],
                str
            )
            or not brief[field].strip()
        ):
            raise ValueError(
                f"{field}는 비어 있지 않은 "
                "문자열이어야 합니다."
            )

    if (
        not isinstance(
            brief["keywords"],
            list
        )
        or not brief["keywords"]
    ):
        raise ValueError(
            "keywords는 비어 있지 않은 "
            "리스트여야 합니다."
        )

    if not all(
        isinstance(
            keyword,
            str
        )
        and keyword.strip()
        for keyword in brief["keywords"]
    ):
        raise ValueError(
            "keywords의 각 항목은 "
            "비어 있지 않은 문자열이어야 합니다."
        )

    if "product_name" in brief:

        if (
            not isinstance(
                brief["product_name"],
                str
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
                str
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
                list
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
                str
            )
            and competitor.strip()
            for competitor
            in brief["competitors"]
        ):
            raise ValueError(
                "competitors의 각 항목은 "
                "비어 있지 않은 문자열이어야 합니다."
            )

    if "notes" in brief:

        if (
            not isinstance(
                brief["notes"],
                str
            )
            or not brief[
                "notes"
            ].strip()
        ):
            raise ValueError(
                "notes는 비어 있지 않은 "
                "문자열이어야 합니다."
            )


# =========================================================
# 브랜드 브리프 불러오기
# =========================================================
def load_brand_brief(
    file_path
):

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
        encoding="utf-8"
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
            "\n브랜드 브리프 JSON 파일 경로를 "
            "입력하세요:\n> "
        ).strip()

        if brief_path:
            break

        print(
            "[오류] 파일 경로는 필수 입력입니다."
        )

    output_path = input(
        "\n출력 폴더 경로를 입력하세요. "
        "(Enter 입력 시 ./output 사용):\n> "
    ).strip()

    if not output_path:
        output_path = "./output"

    try:

        # -------------------------------------------------
        # 브랜드 브리프 읽기
        # -------------------------------------------------
        brief = load_brand_brief(
            brief_path
        )

        # -------------------------------------------------
        # 출력 폴더 생성
        # -------------------------------------------------
        output_dir = Path(
            output_path
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        product_name = brief.get(
            "product_name",
            "제품명 없음"
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

        # -------------------------------------------------
        # 기존 brand_result.json 확인
        # -------------------------------------------------
        result_path = (
            output_dir
            / "brand_result.json"
        )

        if result_path.exists():

            print(
                "\n기존 brand_result.json을 찾았습니다."
            )

            print(
                "기존 결과를 사용하므로 "
                "텍스트 생성 API를 다시 호출하지 않습니다."
            )

            brand_result = (
                load_brand_result(
                    result_path
                )
            )

        else:

            print(
                "\n제품 분석 및 브랜드 "
                "아이덴티티 생성을 시작합니다..."
            )

            client = get_openai_client()

            print(
                "\n브랜드 아이덴티티를 생성합니다..."
            )

            brand_result = (
                generate_brand_identity(
                    client,
                    brief
                )
            )

            print(
                "브랜드 아이덴티티 생성 완료"
            )

            result_path = (
                save_brand_result(
                    brand_result,
                    output_dir
                )
            )

            print(
                "\n브랜드 결과 저장 완료:"
            )

            print(
                result_path
            )

        # -------------------------------------------------
        # 컬러 팔레트 생성
        # -------------------------------------------------
        print(
            "\n컬러 팔레트 이미지를 생성합니다..."
        )

        palette_path = (
            generate_color_palette(
                result_path,
                output_dir
            )
        )

        print(
            "컬러 팔레트 이미지 생성 완료:"
        )

        print(
            palette_path
        )

        # -------------------------------------------------
        # 로고 생성 여부 확인
        # -------------------------------------------------
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

            logo_paths = (
                generate_logo_images(
                    client,
                    brand_result,
                    brief,
                    output_dir
                )
            )

            print(
                "\n로고 이미지 생성 작업이 끝났습니다."
            )

            print(
                f"성공한 로고 수: "
                f"{len(logo_paths)}/3"
            )

        else:

            print(
                "\n로고 이미지 생성을 건너뜁니다."
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