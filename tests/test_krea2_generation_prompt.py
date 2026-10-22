import pytest

from prompting_pot.prompts import (
    FOLDER_CAPTION_STYLES,
    IMAGE_SYSTEM_PROMPTS,
    KREA2_SYSTEM_PROMPT,
    build_folder_caption_request,
    build_image_prompt,
    build_text_prompt,
)


@pytest.mark.parametrize("builder", [build_image_prompt, build_text_prompt])
def test_krea2_generation_has_five_ordered_components_and_copy_line(builder):
    prompt = builder("krea2")
    components = [
        "Medium & Style",
        "Subject & Features",
        "Lighting & Mood",
        "Environment & Background",
        "Framing & Composition",
    ]
    positions = [
        prompt.index(f"{i}. {component}:") for i, component in enumerate(components, 1)
    ]
    assert positions == sorted(positions)
    assert "**Structured Breakdown:**" in prompt
    assert "**Ready-to-Copy Krea2 Prompt:**" in prompt
    assert "one single line of comma-separated descriptive phrases" in prompt
    assert (
        "keyword phrases separated by commas, not a flowing prose paragraph" in prompt
    )
    assert (
        "real photographic capture rather than CGI or a digital composite" not in prompt
    )
    assert "empty changes mean an as-is reconstruction" in prompt
    assert "Do not return JSON or private reasoning" in prompt


def test_text_mode_reuses_new_generation_format_without_image_assumptions():
    assert IMAGE_SYSTEM_PROMPTS["krea2"] == KREA2_SYSTEM_PROMPT
    prompt = build_text_prompt("krea2")
    assert prompt.startswith(KREA2_SYSTEM_PROMPT)
    assert "text-only generation task" in prompt
    assert "no image is supplied" in prompt
    assert "honor an explicit preference" in prompt
    assert "Preserve explicitly requested or clearly readable scene text" in prompt


@pytest.mark.parametrize("style", ["character", "style"])
def test_krea2_lora_prompts_remain_natural_prose_without_generation_breakdown(style):
    prompt, instruction = build_folder_caption_request("krea2", style, "TOK")
    assert prompt == FOLDER_CAPTION_STYLES[style]
    assert "natural, complete English sentences" in prompt
    assert "never comma-separated tags" in prompt
    assert "Structured Breakdown" not in prompt
    assert "Ready-to-Copy" not in prompt
    assert "TOK" in instruction
    if style == "character":
        assert "Do not describe the face" in prompt
        assert "fixed identity features" in prompt
