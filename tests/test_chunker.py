import pytest
from chatterbox.engine.service import split_text_chunks


def test_split_text_chunks_short():
    text = "مرحبا بكم في الاستوديو الصوتي."
    chunks = split_text_chunks(text, max_chunk_len=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_chunks_multiple_sentences():
    text = "الجملة الأولى هنا. الجملة الثانية أطول قليلاً! وهذه الجملة الثالثة للتأكد من التقسيم؟"
    chunks = split_text_chunks(text, max_chunk_len=30)
    assert len(chunks) >= 3
    # All original sentences preserved
    combined = " ".join(chunks)
    assert "الأولى" in combined
    assert "الثانية" in combined
    assert "الثالثة" in combined


def test_split_text_chunks_arabic_punctuation():
    text = "برأيي، إذا دخل العراق هذا المسار الاقتصادي؛ فإن شكل الدولة سيتغير جذرياً: هذا تحليل واضح."
    chunks = split_text_chunks(text, max_chunk_len=35)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 45
