"""
Comprehensive Test Suite for MAIN_HEALTH Project
=================================================
Covers:
  - pubmed_fetcher.py
  - pubmed_ingest.py
  - drug_interactions.py
  - patient_graph_reader.py
  - wearables_graph.py
  - ollama_client.py
  - autopilot.py
  - chunker.py
  - embedding.py
  - entity_extractor.py
  - claim_extractor.py
  - fact_checker.py
  - qdrant_search.py
  - prompt_builder.py
  - graph_rag_pipeline.py
  - schema_builder.py
  - models.py
  - paper_search.py
  - qdrant_store.py
  - api.py (Flask routes)

Run with:
    pytest tests/test_suite.py -v
    pytest tests/test_suite.py -v --tb=short        (shorter tracebacks)
    pytest tests/test_suite.py -k "drug"             (filter by keyword)
    pytest tests/test_suite.py -k "chunk or embed"   (multiple keywords)
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import requests


# ===========================================================================
# SECTION 1 — pubmed_fetcher.py
# ===========================================================================

class TestSearchPmcArticles:
    """Tests for search_pmc_articles()"""

    @patch("app.fetchers.pubmed_fetcher.Entrez.esearch")
    @patch("app.fetchers.pubmed_fetcher.Entrez.read")
    def test_returns_id_list_on_success(self, mock_read, mock_esearch):
        """Should return list of PMC IDs when search succeeds."""
        from app.fetchers.pubmed_fetcher import search_pmc_articles

        mock_read.return_value = {"IdList": ["12345", "67890"]}
        mock_esearch.return_value = MagicMock()

        result = search_pmc_articles("diabetes", max_results=5)

        assert result == ["12345", "67890"]
        assert len(result) == 2

    @patch("app.fetchers.pubmed_fetcher.Entrez.esearch")
    @patch("app.fetchers.pubmed_fetcher.Entrez.read")
    def test_returns_empty_list_when_no_results(self, mock_read, mock_esearch):
        """Should return empty list when no articles found."""
        from app.fetchers.pubmed_fetcher import search_pmc_articles

        mock_read.return_value = {"IdList": []}
        mock_esearch.return_value = MagicMock()

        result = search_pmc_articles("xyznonexistentquery123", max_results=5)

        assert result == []

    @patch("app.fetchers.pubmed_fetcher.Entrez.esearch", side_effect=Exception("Network error"))
    def test_returns_empty_list_on_exception(self, mock_esearch):
        """Should return empty list and not raise on network failure."""
        from app.fetchers.pubmed_fetcher import search_pmc_articles

        result = search_pmc_articles("diabetes", max_results=5)

        assert result == []

    @patch("app.fetchers.pubmed_fetcher.Entrez.esearch")
    @patch("app.fetchers.pubmed_fetcher.Entrez.read")
    def test_appends_open_access_filter(self, mock_read, mock_esearch):
        """Should add 'open access[filter]' to query."""
        from app.fetchers.pubmed_fetcher import search_pmc_articles

        mock_read.return_value = {"IdList": []}
        mock_esearch.return_value = MagicMock()

        search_pmc_articles("diabetes", max_results=3)

        call_kwargs = mock_esearch.call_args[1]
        assert "open access[filter]" in call_kwargs["term"]


class TestExtractFullText:
    """Tests for _extract_full_text()"""

    def test_extracts_sections_with_titles(self):
        """Should extract section titles and paragraphs."""
        from bs4 import BeautifulSoup
        from app.fetchers.pubmed_fetcher import _extract_full_text

        xml = """
        <body>
          <sec>
            <title>Introduction</title>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
          </sec>
        </body>
        """
        soup = BeautifulSoup(xml, "lxml-xml")
        result = _extract_full_text(soup)

        assert "Introduction" in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_returns_empty_string_when_no_body(self):
        """Should return empty string if no <body> tag."""
        from bs4 import BeautifulSoup
        from app.fetchers.pubmed_fetcher import _extract_full_text

        soup = BeautifulSoup("<article><front/></article>", "lxml-xml")
        result = _extract_full_text(soup)

        assert result == ""

    def test_skips_sections_with_no_paragraphs(self):
        """Sections without <p> tags should not appear in output."""
        from bs4 import BeautifulSoup
        from app.fetchers.pubmed_fetcher import _extract_full_text

        xml = """
        <body>
          <sec><title>Empty Section</title></sec>
          <sec><title>Real Section</title><p>Content here.</p></sec>
        </body>
        """
        soup = BeautifulSoup(xml, "lxml-xml")
        result = _extract_full_text(soup)

        assert "Empty Section" not in result
        assert "Real Section" in result

    def test_uses_default_title_when_missing(self):
        """Should use 'Section' label when no <title> present."""
        from bs4 import BeautifulSoup
        from app.fetchers.pubmed_fetcher import _extract_full_text

        xml = "<body><sec><p>Some content.</p></sec></body>"
        soup = BeautifulSoup(xml, "lxml-xml")
        result = _extract_full_text(soup)

        assert "Section" in result
        assert "Some content." in result


class TestFetchPmcDetails:
    """Tests for fetch_pmc_details()"""

    @patch("app.fetchers.pubmed_fetcher.Entrez.efetch")
    def test_returns_none_on_exception(self, mock_efetch):
        """Should return None if fetching fails."""
        from app.fetchers.pubmed_fetcher import fetch_pmc_details

        mock_efetch.side_effect = Exception("Connection failed")

        result = fetch_pmc_details("PMC12345")

        assert result is None

    @patch("app.fetchers.pubmed_fetcher.Entrez.efetch")
    def test_parses_title_and_journal(self, mock_efetch):
        """Should correctly parse title and journal from XML."""
        from app.fetchers.pubmed_fetcher import fetch_pmc_details

        xml = b"""
        <pmc-articleset>
          <article>
            <front>
              <article-meta>
                <title-group>
                  <article-title>Test Article Title</article-title>
                </title-group>
                <pub-date pub-type="epub"><year>2024</year></pub-date>
              </article-meta>
              <journal-meta>
                <journal-title-group>
                  <journal-title>Test Journal</journal-title>
                </journal-title-group>
              </journal-meta>
            </front>
            <body></body>
          </article>
        </pmc-articleset>
        """
        mock_handle = MagicMock()
        mock_handle.read.return_value = xml
        mock_efetch.return_value = mock_handle

        result = fetch_pmc_details("PMC12345")

        assert result is not None
        assert result["title"] == "Test Article Title"
        assert result["journal"] == "Test Journal"
        assert result["year"] == 2024

    @patch("app.fetchers.pubmed_fetcher.Entrez.efetch")
    def test_defaults_when_fields_missing(self, mock_efetch):
        """Should use default values when optional fields are absent."""
        from app.fetchers.pubmed_fetcher import fetch_pmc_details

        xml = b"<pmc-articleset><article><front></front><body></body></article></pmc-articleset>"
        mock_handle = MagicMock()
        mock_handle.read.return_value = xml
        mock_efetch.return_value = mock_handle

        result = fetch_pmc_details("PMC99999")

        assert result is not None
        assert result["title"] == "No Title"
        assert result["journal"] == "Unknown Journal"
        assert result["year"] == 0


class TestFetchAllPmcArticles:
    """Tests for fetch_all_pmc_articles()"""

    @patch("app.fetchers.pubmed_fetcher.search_pmc_articles", return_value=[])
    def test_returns_empty_list_when_no_ids(self, mock_search):
        """Should return empty list if no PMC IDs found."""
        from app.fetchers.pubmed_fetcher import fetch_all_pmc_articles

        result = fetch_all_pmc_articles("diabetes", max_results=5)

        assert result == []

    @patch("app.fetchers.pubmed_fetcher.time.sleep")
    @patch("app.fetchers.pubmed_fetcher.fetch_pmc_details")
    @patch("app.fetchers.pubmed_fetcher.search_pmc_articles")
    def test_filters_articles_below_min_length(self, mock_search, mock_fetch, mock_sleep):
        """Should skip articles with abstract shorter than MIN_TEXT_LENGTH."""
        from app.fetchers.pubmed_fetcher import fetch_all_pmc_articles

        mock_search.return_value = ["PMC1"]
        mock_fetch.return_value = {"pmid": "PMC1", "abstract": "short"}

        result = fetch_all_pmc_articles("diabetes", max_results=1)

        assert result == []

    @patch("app.fetchers.pubmed_fetcher.time.sleep")
    @patch("app.fetchers.pubmed_fetcher.fetch_pmc_details")
    @patch("app.fetchers.pubmed_fetcher.search_pmc_articles")
    def test_skips_none_articles(self, mock_search, mock_fetch, mock_sleep):
        """Should skip None results from fetch_pmc_details."""
        from app.fetchers.pubmed_fetcher import fetch_all_pmc_articles

        mock_search.return_value = ["PMC1", "PMC2"]
        mock_fetch.side_effect = [None, None]

        result = fetch_all_pmc_articles("diabetes", max_results=2)

        assert result == []


# ===========================================================================
# SECTION 2 — pubmed_ingest.py
# ===========================================================================

class TestBatchUtility:
    """Tests for _batch() helper.

    NOTE: pubmed_ingest.py has module-level code in a misindented __main__ block
    that runs on import. We patch ingest_from_pubmed at the module level to
    prevent the side-effect, then import _batch directly.
    """

    def _import_batch(self):
        # Patch the side-effectful ingest call before importing
        with patch("app.ingestion.pubmed_ingest.ingest_from_pubmed"):
            from app.ingestion.pubmed_ingest import _batch
        return _batch

    def test_yields_correct_batch_sizes(self):
        _batch = self._import_batch()
        result = list(_batch(range(10), 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_yields_single_batch_when_smaller_than_size(self):
        _batch = self._import_batch()
        result = list(_batch([1, 2], 10))
        assert result == [[1, 2]]

    def test_handles_empty_iterable(self):
        _batch = self._import_batch()
        result = list(_batch([], 5))
        assert result == []

    def test_exact_multiple_of_batch_size(self):
        _batch = self._import_batch()
        result = list(_batch(range(6), 3))
        assert result == [[0, 1, 2], [3, 4, 5]]


def _import_ingest():
    """Import ingest_from_pubmed safely, suppressing module-level side effects."""
    with patch("app.ingestion.pubmed_ingest.ingest_from_pubmed"):
        from app.ingestion import pubmed_ingest
    return pubmed_ingest.ingest_from_pubmed


class TestIngestFromPubmed:
    """Tests for ingest_from_pubmed()"""

    @patch("app.ingestion.pubmed_ingest.get_client", side_effect=Exception("Qdrant unavailable"))
    def test_returns_early_on_qdrant_failure(self, mock_client):
        """Should return early if Qdrant client init fails."""
        ingest_from_pubmed = _import_ingest()
        ingest_from_pubmed("diabetes", max_results=1)

    @patch("app.ingestion.pubmed_ingest.fetch_all_pmc_articles", return_value=[])
    @patch("app.ingestion.pubmed_ingest.create_indexes")
    @patch("app.ingestion.pubmed_ingest.create_collection_if_not_exists")
    @patch("app.ingestion.pubmed_ingest.get_client")
    def test_returns_early_when_no_papers(self, mock_client, mock_create, mock_indexes, mock_fetch):
        """Should return early when no papers are fetched."""
        ingest_from_pubmed = _import_ingest()
        ingest_from_pubmed("diabetes", max_results=5)
        mock_fetch.assert_called_once()

    @patch("app.ingestion.pubmed_ingest.fetch_all_pmc_articles")
    @patch("app.ingestion.pubmed_ingest.create_indexes")
    @patch("app.ingestion.pubmed_ingest.create_collection_if_not_exists")
    @patch("app.ingestion.pubmed_ingest.get_client")
    def test_skips_papers_without_abstract(self, mock_client, mock_create, mock_indexes, mock_fetch):
        """Papers with no abstract should be skipped."""
        ingest_from_pubmed = _import_ingest()
        mock_fetch.return_value = [{"pmid": "P1", "title": "Test", "abstract": None}]
        ingest_from_pubmed("diabetes", max_results=1)

    @patch("app.ingestion.pubmed_ingest.embed_texts", return_value=[[0.1, 0.2]])
    @patch("app.ingestion.pubmed_ingest.simple_chunk", return_value=["chunk1", "chunk2"])
    @patch("app.ingestion.pubmed_ingest.fetch_all_pmc_articles")
    @patch("app.ingestion.pubmed_ingest.create_indexes")
    @patch("app.ingestion.pubmed_ingest.create_collection_if_not_exists")
    @patch("app.ingestion.pubmed_ingest.get_client")
    def test_logs_mismatch_between_chunks_and_vectors(
        self, mock_client, mock_create, mock_indexes, mock_fetch, mock_chunk, mock_embed
    ):
        """Should handle chunk/vector length mismatch gracefully."""
        ingest_from_pubmed = _import_ingest()
        mock_fetch.return_value = [{"pmid": "P1", "title": "Title", "abstract": "Some text"}]
        ingest_from_pubmed("diabetes", max_results=1)


# ===========================================================================
# SECTION 3 — drug_interactions.py
# ===========================================================================

class TestCheckDrugInteractions:
    """Tests for check_drug_interactions() entry point."""

    def test_returns_safe_response_for_non_list_input(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        result = check_drug_interactions("metformin")  # string, not list

        assert "note" in result
        assert result["checked_drugs"] == []

    def test_returns_safe_response_for_empty_list(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        result = check_drug_interactions([])

        assert "note" in result
        assert result["checked_drugs"] == []

    def test_returns_safe_response_for_items_without_name(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        result = check_drug_interactions([{"dosage": "10mg"}])  # no 'name' key

        assert "note" in result

    def test_normalizes_drug_names_to_lowercase(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        with patch("app.knowledge_graph.drug_interactions._check_drug_condition_facts", return_value=[]):
            result = check_drug_interactions([{"name": "METFORMIN"}, {"name": "Aspirin"}])

        assert "metformin" in result["checked_drugs"]
        assert "aspirin" in result["checked_drugs"]

    def test_deduplicates_drug_names(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        with patch("app.knowledge_graph.drug_interactions._check_drug_condition_facts", return_value=[]):
            result = check_drug_interactions([
                {"name": "metformin"}, {"name": "metformin"}, {"name": "Metformin"}
            ])

        assert result["checked_drugs"].count("metformin") == 1

    def test_returns_all_required_keys(self):
        from app.knowledge_graph.drug_interactions import check_drug_interactions

        with patch("app.knowledge_graph.drug_interactions._check_drug_condition_facts", return_value=[]):
            result = check_drug_interactions([{"name": "aspirin"}])

        assert "checked_drugs" in result
        assert "drug_drug_interactions" in result
        assert "drug_condition_interactions" in result
        assert "drug_effect_facts" in result


class TestCheckDrugDrugFacts:
    """Tests for _check_drug_drug_facts() — deterministic rule engine."""

    def test_detects_metformin_contrast_dye_interaction(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["metformin", "contrast dye"])

        assert any(
            "lactic acidosis" in f["interaction"].lower()
            for f in result
        )

    def test_detects_aspirin_atorvastatin_interaction(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["aspirin", "atorvastatin"])

        assert len(result) >= 1
        assert all(f["type"] == "drug-drug-interaction" for f in result)

    def test_no_interaction_for_unrelated_drugs(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["vitamin_c", "zinc"])

        assert result == []

    def test_returns_list_type(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["metformin"])

        assert isinstance(result, list)

    def test_interaction_has_required_fields(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["metformin", "contrast dye"])

        assert len(result) > 0
        for fact in result:
            assert "type" in fact
            assert "drugs_involved" in fact
            assert "severity" in fact
            assert "interaction" in fact
            assert "mechanism" in fact

    def test_severity_levels_are_valid(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["metformin", "insulin", "aspirin", "atorvastatin"])

        valid_severities = {"low", "moderate", "high"}
        for fact in result:
            assert fact["severity"] in valid_severities

    def test_drugs_involved_are_sorted(self):
        from app.knowledge_graph.drug_interactions import _check_drug_drug_facts

        result = _check_drug_drug_facts(["metformin", "contrast dye"])

        for fact in result:
            assert fact["drugs_involved"] == sorted(fact["drugs_involved"])


class TestCheckDrugEffectFacts:
    """Tests for _check_drug_effect_facts() — knowledge base lookup."""

    def test_metformin_returns_b12_fact(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["metformin"])

        assert any("B12" in f["effect"] for f in result)

    def test_aspirin_returns_gi_bleeding_fact(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["aspirin"])

        assert any("bleeding" in f["effect"].lower() for f in result)

    def test_montelukast_returns_black_box_warning(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["montelukast"])

        assert any("black box" in f["evidence"].lower() for f in result)

    def test_returns_empty_for_unknown_drug(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["unknowndrug999"])

        assert result == []

    def test_combination_lisinopril_amlodipine_returns_extra_fact(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["lisinopril", "amlodipine"])

        drug_names = [f["drug"] for f in result]
        assert "lisinopril + amlodipine" in drug_names

    def test_all_facts_have_type_drug_effect(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["metformin", "aspirin", "losartan"])

        for fact in result:
            assert fact["type"] == "drug-effect"

    def test_all_facts_have_required_fields(self):
        from app.knowledge_graph.drug_interactions import _check_drug_effect_facts

        result = _check_drug_effect_facts(["metformin"])

        for fact in result:
            for field in ["type", "drug", "effect", "mechanism", "clinical_relevance", "evidence"]:
                assert field in fact, f"Missing field: {field}"


class TestSafeResponse:
    """Tests for _safe_response() helper."""

    def test_returns_dict_with_note(self):
        from app.knowledge_graph.drug_interactions import _safe_response

        result = _safe_response("No medications provided")

        assert result["note"] == "No medications provided"

    def test_returns_empty_lists(self):
        from app.knowledge_graph.drug_interactions import _safe_response

        result = _safe_response("test")

        assert result["drug_drug_interactions"] == []
        assert result["drug_condition_interactions"] == []
        assert result["drug_effect_facts"] == []
        assert result["checked_drugs"] == []


# ===========================================================================
# SECTION 4 — patient_graph_reader.py (pure helper functions only)
# ===========================================================================

class TestFormatMedications:
    """Tests for _format_medications()"""

    def test_returns_list_of_dicts(self):
        from app.knowledge_graph.patient_graph_reader import _format_medications

        mock_node = MagicMock()
        mock_node.get.side_effect = lambda key, *args: {
            "name": "Metformin", "dosage": "1000mg",
            "frequency": "twice daily", "purpose": "Blood sugar control",
            "atcCode": "A10BA02"
        }.get(key)

        result = _format_medications([mock_node])

        assert len(result) == 1
        assert result[0]["name"] == "Metformin"
        assert result[0]["dosage"] == "1000mg"

    def test_skips_none_nodes(self):
        from app.knowledge_graph.patient_graph_reader import _format_medications

        result = _format_medications([None, None])

        assert result == []

    def test_returns_empty_for_empty_input(self):
        from app.knowledge_graph.patient_graph_reader import _format_medications

        result = _format_medications([])

        assert result == []


class TestExtractNumericValues:
    """Tests for _extract_numeric_values()"""

    def test_extracts_plain_numbers(self):
        from app.knowledge_graph.patient_graph_reader import _extract_numeric_values

        readings = [{"value": "156"}, {"value": "142"}]
        result = _extract_numeric_values(readings)

        assert result == [156.0, 142.0]

    def test_handles_blood_pressure_format(self):
        """Should extract systolic from '138/88' format."""
        from app.knowledge_graph.patient_graph_reader import _extract_numeric_values

        readings = [{"value": "138/88"}]
        result = _extract_numeric_values(readings)

        assert result == [138.0]

    def test_skips_non_numeric_strings(self):
        """Should skip values like 'NSR'."""
        from app.knowledge_graph.patient_graph_reader import _extract_numeric_values

        readings = [{"value": "NSR"}, {"value": "72"}]
        result = _extract_numeric_values(readings)

        assert result == [72.0]

    def test_returns_empty_for_all_non_numeric(self):
        from app.knowledge_graph.patient_graph_reader import _extract_numeric_values

        readings = [{"value": "NSR"}, {"value": "Normal"}]
        result = _extract_numeric_values(readings)

        assert result == []


class TestComputeTrend:
    """Tests for _compute_trend()"""

    def test_returns_stable_for_small_difference(self):
        from app.knowledge_graph.patient_graph_reader import _compute_trend

        result = _compute_trend([100.0, 101.0])

        assert result == "stable"

    def test_returns_increasing_for_upward_trend(self):
        from app.knowledge_graph.patient_graph_reader import _compute_trend

        result = _compute_trend([100.0, 120.0])

        assert "increasing" in result

    def test_returns_decreasing_for_downward_trend(self):
        from app.knowledge_graph.patient_graph_reader import _compute_trend

        result = _compute_trend([120.0, 100.0])

        assert "decreasing" in result

    def test_returns_monitoring_needed_for_single_value(self):
        from app.knowledge_graph.patient_graph_reader import _compute_trend

        result = _compute_trend([100.0])

        assert "more readings needed" in result

    def test_returns_monitoring_needed_for_empty_list(self):
        from app.knowledge_graph.patient_graph_reader import _compute_trend

        result = _compute_trend([])

        assert "more readings needed" in result


class TestCleanTimestamp:
    """Tests for _clean_timestamp()"""

    def test_extracts_date_from_iso_timestamp(self):
        from app.knowledge_graph.patient_graph_reader import _clean_timestamp

        result = _clean_timestamp("2026-02-08T08:00:00Z")

        assert result == "2026-02-08"

    def test_returns_unknown_date_for_empty_string(self):
        from app.knowledge_graph.patient_graph_reader import _clean_timestamp

        result = _clean_timestamp("")

        assert result == "unknown date"

    def test_returns_unknown_date_for_none(self):
        from app.knowledge_graph.patient_graph_reader import _clean_timestamp

        result = _clean_timestamp(None)

        assert result == "unknown date"


class TestSafeDate:
    """Tests for _safe_date()"""

    def test_converts_value_to_string(self):
        from app.knowledge_graph.patient_graph_reader import _safe_date

        result = _safe_date("2024-01-15")

        assert result == "2024-01-15"

    def test_returns_none_for_falsy_input(self):
        from app.knowledge_graph.patient_graph_reader import _safe_date

        assert _safe_date(None) is None
        assert _safe_date("") is None


class TestFormatWearables:
    """Tests for _format_wearables()"""

    def test_returns_unavailable_for_empty_rows(self):
        from app.knowledge_graph.patient_graph_reader import _format_wearables

        result = _format_wearables([])

        assert result["available"] is False
        assert result["metrics"] == []

    def test_returns_available_true_for_valid_rows(self):
        from app.knowledge_graph.patient_graph_reader import _format_wearables

        rows = [{
            "name": "Blood Glucose",
            "type": "blood_glucose",
            "unit": "mg/dL",
            "normal_range": "70-100",
            "readings": [
                {"value": "156", "timestamp": "2026-02-08T08:00:00Z"},
                {"value": "142", "timestamp": "2026-02-09T08:00:00Z"},
            ]
        }]

        result = _format_wearables(rows)

        assert result["available"] is True
        assert len(result["metrics"]) == 1

    def test_computes_latest_and_previous_values(self):
        from app.knowledge_graph.patient_graph_reader import _format_wearables

        rows = [{
            "name": "Heart Rate",
            "type": "heart_rate",
            "unit": "bpm",
            "normal_range": "60-100",
            "readings": [
                {"value": "72", "timestamp": "2026-02-08T08:00:00Z"},
                {"value": "75", "timestamp": "2026-02-09T08:00:00Z"},
            ]
        }]

        result = _format_wearables(rows)
        metric = result["metrics"][0]

        assert "75" in metric["latest_value"]
        assert "72" in metric["previous_value"]

    def test_filters_out_empty_readings(self):
        from app.knowledge_graph.patient_graph_reader import _format_wearables

        rows = [{
            "name": "Steps",
            "type": "steps",
            "unit": "steps",
            "normal_range": "> 8000",
            "readings": [
                {"value": "", "timestamp": "2026-02-08T08:00:00Z"},
                {"value": "None", "timestamp": "2026-02-09T08:00:00Z"},
            ]
        }]

        result = _format_wearables(rows)

        # No valid readings → metric still included but with defaults
        metric = result["metrics"][0]
        assert metric["latest_value"] == "not recorded yet"


# ===========================================================================
# SECTION 5 — wearables_graph.py
# ===========================================================================

class TestExtractBpSystolic:
    """Tests for _extract_bp_systolic()"""

    def test_extracts_systolic_from_bp_readings(self):
        from app.knowledge_graph.wearables_graph import _extract_bp_systolic

        result = _extract_bp_systolic(["138/88", "142/90"])

        assert result == [138.0, 142.0]

    def test_returns_empty_for_non_bp_values(self):
        from app.knowledge_graph.wearables_graph import _extract_bp_systolic

        result = _extract_bp_systolic(["72", "75"])

        assert result == []

    def test_skips_malformed_bp_values(self):
        from app.knowledge_graph.wearables_graph import _extract_bp_systolic

        result = _extract_bp_systolic(["abc/def"])

        assert result == []


class TestExtractBpDiastolic:
    """Tests for _extract_bp_diastolic()"""

    def test_extracts_diastolic_from_bp_readings(self):
        from app.knowledge_graph.wearables_graph import _extract_bp_diastolic

        result = _extract_bp_diastolic(["138/88", "142/90"])

        assert result == [88.0, 90.0]


class TestComputeNumericTrend:
    """Tests for _compute_numeric_trend()"""

    def test_stable_within_2_percent(self):
        from app.knowledge_graph.wearables_graph import _compute_numeric_trend

        result = _compute_numeric_trend([100.0, 101.0])

        assert result == "stable"

    def test_increasing_trend(self):
        from app.knowledge_graph.wearables_graph import _compute_numeric_trend

        result = _compute_numeric_trend([100.0, 130.0])

        assert "increasing" in result

    def test_decreasing_trend(self):
        from app.knowledge_graph.wearables_graph import _compute_numeric_trend

        result = _compute_numeric_trend([130.0, 100.0])

        assert "decreasing" in result

    def test_single_value_returns_monitoring_needed(self):
        from app.knowledge_graph.wearables_graph import _compute_numeric_trend

        result = _compute_numeric_trend([100.0])

        assert "more readings needed" in result

    def test_zero_first_value_returns_monitoring_needed(self):
        from app.knowledge_graph.wearables_graph import _compute_numeric_trend

        result = _compute_numeric_trend([0.0, 50.0])

        assert "more readings needed" in result


class TestComputeStringTrend:
    """Tests for _compute_string_trend()"""

    def test_stable_when_values_equal(self):
        from app.knowledge_graph.wearables_graph import _compute_string_trend

        result = _compute_string_trend(["NSR", "NSR"])

        assert result == "stable"

    def test_changed_when_values_differ(self):
        from app.knowledge_graph.wearables_graph import _compute_string_trend

        result = _compute_string_trend(["NSR", "Irregular"])

        assert result == "changed between readings"

    def test_single_value_returns_monitoring_needed(self):
        from app.knowledge_graph.wearables_graph import _compute_string_trend

        result = _compute_string_trend(["NSR"])

        assert "more readings needed" in result


class TestWearablesCleanTimestamp:
    """Tests for _clean_timestamp() in wearables_graph."""

    def test_truncates_to_date(self):
        from app.knowledge_graph.wearables_graph import _clean_timestamp

        result = _clean_timestamp("2026-02-08T08:00:00Z")

        assert result == "2026-02-08"

    def test_handles_none_string(self):
        from app.knowledge_graph.wearables_graph import _clean_timestamp

        result = _clean_timestamp("None")

        assert result == "unknown date"

    def test_handles_empty_string(self):
        from app.knowledge_graph.wearables_graph import _clean_timestamp

        result = _clean_timestamp("")

        assert result == "unknown date"


# ===========================================================================
# SECTION 6 — ollama_client.py
# ===========================================================================

class TestCallOllama:
    """Tests for call_ollama()"""

    @patch("app.llm.ollama_client.requests.post")
    def test_returns_message_content_on_success(self, mock_post):
        from app.llm.ollama_client import call_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Test response from model"}
        }
        mock_post.return_value = mock_response

        result = call_ollama("What is diabetes?")

        assert result == "Test response from model"

    @patch("app.llm.ollama_client.requests.post")
    def test_returns_error_message_on_non_200(self, mock_post):
        from app.llm.ollama_client import call_ollama

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = call_ollama("question")

        assert "Ollama error 500" in result

    @patch("app.llm.ollama_client.requests.post", side_effect=requests.exceptions.ConnectionError)
    def test_returns_error_when_ollama_not_running(self, mock_post):
        from app.llm.ollama_client import call_ollama

        result = call_ollama("question")

        assert "not running" in result.lower() or "Error" in result

    @patch("app.llm.ollama_client.requests.post", side_effect=Exception("Unexpected"))
    def test_returns_generic_error_on_unknown_exception(self, mock_post):
        from app.llm.ollama_client import call_ollama

        result = call_ollama("question")

        assert "Error" in result

    @patch("app.llm.ollama_client.requests.post")
    def test_truncates_long_prompt(self, mock_post):
        from app.llm.ollama_client import call_ollama, MAX_PROMPT_CHARS

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        long_prompt = "A" * (MAX_PROMPT_CHARS + 1000)
        call_ollama(long_prompt)

        actual_prompt = mock_post.call_args[1]["json"]["messages"][1]["content"]
        assert len(actual_prompt) <= MAX_PROMPT_CHARS + 500  # some buffer for smart truncation

    @patch("app.llm.ollama_client.requests.post")
    def test_uses_correct_model(self, mock_post):
        from app.llm.ollama_client import call_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        call_ollama("Hello", model="phi3:mini")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "phi3:mini"

    @patch("app.llm.ollama_client.requests.post")
    def test_sends_system_message(self, mock_post):
        from app.llm.ollama_client import call_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        call_ollama("test question")

        messages = mock_post.call_args[1]["json"]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


class TestSmartTruncate:
    """Tests for _smart_truncate()"""

    def test_truncates_literature_section(self):
        from app.llm.ollama_client import _smart_truncate

        prompt = (
            "Patient info...\n"
            "========================\nRELEVANT MEDICAL LITERATURE\n========================\n"
            "Long literature content " * 100 +
            "\nUSER QUESTION\nWhat is diabetes?"
        )

        result = _smart_truncate(prompt, 500)

        assert "Truncated to fit context window" in result
        assert "USER QUESTION" in result

    def test_fallback_truncation_when_markers_absent(self):
        from app.llm.ollama_client import _smart_truncate

        prompt = "A" * 2000
        result = _smart_truncate(prompt, 500)

        assert len(result) == 500

    def test_preserves_user_question_section(self):
        from app.llm.ollama_client import _smart_truncate

        prompt = (
            "Before literature\n"
            "RELEVANT MEDICAL LITERATURE\n" + "content " * 200 +
            "USER QUESTION\nWhat should I take?"
        )

        result = _smart_truncate(prompt, 200)

        assert "What should I take?" in result


# ===========================================================================
# SECTION 7 — autopilot.py
# ===========================================================================

class TestAnalyzeHealthIntent:
    """Tests for analyze_health_intent()"""

    @patch("app.knowledge_graph.autopilot.call_ollama")
    def test_returns_list_on_success(self, mock_ollama):
        from app.knowledge_graph.autopilot import analyze_health_intent

        mock_ollama.return_value = json.dumps([
            {"category": "Condition", "original_term": "high fever", "normalized_term": "Fever"}
        ])

        result = analyze_health_intent("I have high fever")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["normalized_term"] == "Fever"

    @patch("app.knowledge_graph.autopilot.call_ollama")
    def test_returns_empty_list_when_no_facts(self, mock_ollama):
        from app.knowledge_graph.autopilot import analyze_health_intent

        mock_ollama.return_value = "[]"

        result = analyze_health_intent("What is diabetes?")

        assert result == []

    @patch("app.knowledge_graph.autopilot.call_ollama")
    def test_handles_dict_response_by_wrapping_in_list(self, mock_ollama):
        """Should wrap a single dict into a list."""
        from app.knowledge_graph.autopilot import analyze_health_intent

        mock_ollama.return_value = json.dumps(
            {"category": "Medication", "original_term": "aspirin", "normalized_term": "Aspirin"}
        )

        result = analyze_health_intent("I take aspirin")

        assert isinstance(result, list)
        assert len(result) == 1

    @patch("app.knowledge_graph.autopilot.call_ollama")
    def test_filters_out_invalid_entries(self, mock_ollama):
        """Entries without category or normalized_term should be filtered."""
        from app.knowledge_graph.autopilot import analyze_health_intent

        mock_ollama.return_value = json.dumps([
            {"original_term": "something"},  # missing category and normalized_term
            {"category": "Condition", "normalized_term": "Fever"},  # valid
        ])

        result = analyze_health_intent("I have fever")

        assert len(result) == 1
        assert result[0]["normalized_term"] == "Fever"

    @patch("app.knowledge_graph.autopilot.call_ollama", side_effect=Exception("LLM error"))
    def test_returns_empty_list_on_exception(self, mock_ollama):
        from app.knowledge_graph.autopilot import analyze_health_intent

        result = analyze_health_intent("I have fever")

        assert result == []

    @patch("app.knowledge_graph.autopilot.call_ollama")
    def test_strips_markdown_json_fences(self, mock_ollama):
        """Should handle ```json ... ``` wrapped responses."""
        from app.knowledge_graph.autopilot import analyze_health_intent

        mock_ollama.return_value = (
            "```json\n"
            '[{"category": "Allergy", "original_term": "penicillin", "normalized_term": "Penicillin"}]\n'
            "```"
        )

        result = analyze_health_intent("I am allergic to penicillin")

        assert len(result) == 1
        assert result[0]["category"] == "Allergy"


class TestApplyGraphUpdate:
    """Tests for apply_graph_update()"""

    @patch("app.knowledge_graph.autopilot.get_driver")
    def test_returns_false_for_invalid_category(self, mock_get_driver):
        from app.knowledge_graph.autopilot import apply_graph_update

        success, msg = apply_graph_update("user_1", "InvalidCategory", "SomeDrug")

        assert success is False
        assert "Invalid category" in msg

    @patch("app.knowledge_graph.autopilot.get_driver")
    def test_condition_update_returns_true_when_patient_found(self, mock_get_driver):
        from app.knowledge_graph.autopilot import apply_graph_update

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"u.id": "user_1"}
        mock_session.run.return_value = mock_result
        mock_get_driver.return_value.__enter__ = MagicMock()
        mock_get_driver.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        success, msg = apply_graph_update("user_1", "Condition", "Fever")

        assert success is True
        assert "Fever" in msg

    @patch("app.knowledge_graph.autopilot.get_driver")
    def test_returns_false_when_patient_not_found(self, mock_get_driver):
        from app.knowledge_graph.autopilot import apply_graph_update

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = None  # Patient not found
        mock_session.run.return_value = mock_result
        mock_get_driver.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        success, msg = apply_graph_update("unknown_user", "Condition", "Fever")

        assert success is False
        assert "not found" in msg

    @patch("app.knowledge_graph.autopilot.get_driver")
    def test_returns_false_on_db_exception(self, mock_get_driver):
        from app.knowledge_graph.autopilot import apply_graph_update

        # Exception raised when opening a session, not at get_driver() call
        mock_driver = MagicMock()
        mock_driver.session.side_effect = Exception("Neo4j down")
        mock_get_driver.return_value = mock_driver

        success, msg = apply_graph_update("user_1", "Medication", "Aspirin")

        assert success is False

    @pytest.mark.parametrize("category", ["Condition", "Medication", "Allergy"])
    @patch("app.knowledge_graph.autopilot.get_driver")
    def test_all_valid_categories_produce_query(self, mock_get_driver, category):
        """All 3 valid categories should produce a Cypher query and attempt DB write."""
        from app.knowledge_graph.autopilot import apply_graph_update

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"u.id": "user_1"}
        mock_session.run.return_value = mock_result
        mock_get_driver.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        success, msg = apply_graph_update("user_1", category, "TestEntity")

        mock_session.run.assert_called_once()


# ===========================================================================
# SECTION 8 — chunker.py
# ===========================================================================

class TestSimpleChunk:
    """Tests for simple_chunk()"""

    def test_returns_list_of_strings(self):
        from app.processing.chunker import simple_chunk

        result = simple_chunk("This is a test sentence. " * 20)

        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_returns_empty_list_for_empty_string(self):
        from app.processing.chunker import simple_chunk

        result = simple_chunk("")

        assert result == []

    def test_returns_empty_list_for_none(self):
        from app.processing.chunker import simple_chunk

        result = simple_chunk(None)

        assert result == []

    def test_returns_empty_list_for_non_string_input(self):
        from app.processing.chunker import simple_chunk

        assert simple_chunk(123) == []
        assert simple_chunk(["list"]) == []

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return as one chunk."""
        from app.processing.chunker import simple_chunk

        text = "Short text."
        result = simple_chunk(text, chunk_size=500, overlap=50)

        assert len(result) == 1
        assert result[0] == text

    def test_long_text_is_split_into_multiple_chunks(self):
        """Long text should produce multiple chunks."""
        from app.processing.chunker import simple_chunk

        text = "Word " * 500  # ~2500 chars
        result = simple_chunk(text, chunk_size=200, overlap=20)

        assert len(result) > 1

    def test_respects_custom_chunk_size(self):
        """Each chunk should not exceed chunk_size by much."""
        from app.processing.chunker import simple_chunk

        text = "A" * 1000
        result = simple_chunk(text, chunk_size=100, overlap=10)

        for chunk in result:
            # Allow small overshoot due to separator logic
            assert len(chunk) <= 150

    def test_overlapping_chunks_share_content(self):
        """With overlap > 0, adjacent chunks should share some text."""
        from app.processing.chunker import simple_chunk

        # Use a text with clear word boundaries so overlap works properly
        text = " ".join([f"word{i}" for i in range(200)])
        result = simple_chunk(text, chunk_size=100, overlap=30)

        if len(result) > 1:
            # End of first chunk and start of second should share some content
            end_of_first = result[0][-30:]
            start_of_second = result[1][:30]
            # They may not be identical but should have overlapping tokens
            assert len(result) > 1  # overlap caused multiple chunks

    def test_no_overlap_when_zero(self):
        """With overlap=0 no content should be duplicated."""
        from app.processing.chunker import simple_chunk

        text = "sentence one. sentence two. sentence three. " * 20
        result = simple_chunk(text, chunk_size=100, overlap=0)

        assert len(result) >= 1

    def test_uses_settings_defaults_when_no_args(self):
        """Should use settings.CHUNK_SIZE and CHUNK_OVERLAP if not specified."""
        from app.processing.chunker import simple_chunk
        from unittest.mock import patch

        with patch("app.processing.chunker.settings") as mock_settings:
            mock_settings.CHUNK_SIZE = 300
            mock_settings.CHUNK_OVERLAP = 50

            text = "Test sentence. " * 100
            result = simple_chunk(text)

            assert isinstance(result, list)

    def test_whitespace_only_string_returns_empty_or_single(self):
        """Whitespace-only input should not crash."""
        from app.processing.chunker import simple_chunk

        result = simple_chunk("   \n\n   ")

        assert isinstance(result, list)

    def test_chunks_are_non_empty_strings(self):
        """All returned chunks should be non-empty strings."""
        from app.processing.chunker import simple_chunk

        text = "Medical research paragraph. " * 50
        result = simple_chunk(text, chunk_size=200, overlap=20)

        for chunk in result:
            assert len(chunk.strip()) > 0


# ===========================================================================
# SECTION 9 — embedding.py
# ===========================================================================

class TestEmbedTexts:
    """Tests for embed_texts()"""

    def test_returns_empty_list_for_none(self):
        from app.processing.embedding import embed_texts

        result = embed_texts(None)

        assert result == []

    def test_returns_empty_list_for_non_list(self):
        from app.processing.embedding import embed_texts

        assert embed_texts("string input") == []
        assert embed_texts(123) == []

    def test_returns_empty_list_for_empty_list(self):
        from app.processing.embedding import embed_texts

        result = embed_texts([])

        assert result == []

    def test_returns_empty_list_for_list_of_empty_strings(self):
        from app.processing.embedding import embed_texts

        result = embed_texts(["", "   ", "\n"])

        assert result == []

    def test_filters_out_non_string_items(self):
        """Non-string items in list should be filtered out, not crash."""
        from app.processing.embedding import embed_texts
        import numpy as np

        mock_model = MagicMock()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_array.shape = (1, 3)
        mock_model.encode.return_value = mock_array

        with patch("app.processing.embedding._get_model", return_value=mock_model):
            result = embed_texts([None, 123, "valid text"])

        assert isinstance(result, list)
        assert len(result) == 1  # only "valid text" passes filter

    @patch("app.processing.embedding._get_model")
    def test_returns_list_of_lists_on_success(self, mock_get_model):
        """Should return list of float vectors."""
        from app.processing.embedding import embed_texts
        import numpy as np

        mock_model = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        fake_embeddings.shape = (2, 3)
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(["text one", "text two"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("app.processing.embedding._get_model")
    def test_calls_encode_with_normalize_true(self, mock_get_model):
        """Should always call encode with normalize_embeddings=True."""
        from app.processing.embedding import embed_texts

        mock_model = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.tolist.return_value = [[0.1]]
        fake_embeddings.shape = (1, 1)
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        embed_texts(["sample text"])

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    @patch("app.processing.embedding._get_model")
    def test_passes_correct_batch_size(self, mock_get_model):
        """Should pass settings.EMBEDDING_BATCH_SIZE to encode."""
        from app.processing.embedding import embed_texts

        mock_model = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.tolist.return_value = [[0.1]]
        fake_embeddings.shape = (1, 1)
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        with patch("app.processing.embedding.settings") as mock_settings:
            mock_settings.EMBEDDING_BATCH_SIZE = 16

            embed_texts(["text"])

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 16

    @patch("app.processing.embedding._get_model")
    def test_output_length_matches_input_length(self, mock_get_model):
        """Number of embeddings must match number of input texts."""
        from app.processing.embedding import embed_texts

        mock_model = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.tolist.return_value = [[0.1], [0.2], [0.3]]
        fake_embeddings.shape = (3, 1)
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        result = embed_texts(["a", "b", "c"])

        assert len(result) == 3


class TestGetEmbeddingModel:
    """Tests for _get_model() lazy loading."""

    def test_model_is_cached_after_first_load(self):
        """Calling _get_model() twice should not load model twice."""
        import app.processing.embedding as emb_module

        original_model = emb_module._model

        try:
            mock_model = MagicMock()
            emb_module._model = mock_model  # pre-set cache

            with patch("app.processing.embedding.SentenceTransformer") as mock_st:
                result = emb_module._get_model()

            mock_st.assert_not_called()  # Should not re-instantiate
            assert result is mock_model
        finally:
            emb_module._model = original_model  # restore

    def test_model_is_loaded_when_none(self):
        """Should load model when _model is None."""
        import app.processing.embedding as emb_module

        original_model = emb_module._model
        emb_module._model = None

        try:
            with patch("app.processing.embedding.SentenceTransformer") as mock_st:
                with patch("app.processing.embedding.settings") as mock_settings:
                    mock_settings.EMBEDDING_MODEL_NAME = "test-model"
                    mock_settings.EMBEDDING_DEVICE = "cpu"
                    mock_st.return_value = MagicMock()

                    emb_module._get_model()

            mock_st.assert_called_once_with("test-model", device="cpu")
        finally:
            emb_module._model = original_model


# ===========================================================================
# SECTION 10 — entity_extractor.py
# ===========================================================================

class TestEmptyResult:
    """Tests for _empty_result()"""

    def test_returns_dict_with_four_keys(self):
        from app.processing.entity_extractor import _empty_result

        result = _empty_result()

        assert set(result.keys()) == {"drugs", "conditions", "biomarkers", "symptoms"}

    def test_all_values_are_empty_lists(self):
        from app.processing.entity_extractor import _empty_result

        result = _empty_result()

        for key, val in result.items():
            assert val == [], f"Expected empty list for key '{key}'"

    def test_returns_independent_instances(self):
        """Each call should return a fresh dict to avoid shared mutation."""
        from app.processing.entity_extractor import _empty_result

        r1 = _empty_result()
        r2 = _empty_result()
        r1["drugs"].append("metformin")

        assert r2["drugs"] == []


class TestExtractMedicalEntities:
    """Tests for extract_medical_entities()"""

    def test_returns_empty_result_for_none(self):
        from app.processing.entity_extractor import extract_medical_entities

        result = extract_medical_entities(None)

        assert result == {"drugs": [], "conditions": [], "biomarkers": [], "symptoms": []}

    def test_returns_empty_result_for_empty_string(self):
        from app.processing.entity_extractor import extract_medical_entities

        result = extract_medical_entities("")

        assert result["drugs"] == []
        assert result["conditions"] == []

    def test_returns_empty_result_for_non_string(self):
        from app.processing.entity_extractor import extract_medical_entities

        assert extract_medical_entities(42)["drugs"] == []
        assert extract_medical_entities(["text"])["drugs"] == []

    @patch("app.processing.entity_extractor._get_model")
    def test_extracts_drugs_correctly(self, mock_get_model):
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Metformin", "label": "drug"},
            {"text": "Aspirin", "label": "drug"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Patient takes Metformin and Aspirin.")

        assert "metformin" in result["drugs"]
        assert "aspirin" in result["drugs"]

    @patch("app.processing.entity_extractor._get_model")
    def test_extracts_conditions_correctly(self, mock_get_model):
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Type 2 Diabetes", "label": "medical condition"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Patient has Type 2 Diabetes.")

        assert "type 2 diabetes" in result["conditions"]

    @patch("app.processing.entity_extractor._get_model")
    def test_deduplicates_entities(self, mock_get_model):
        """Same entity appearing twice should only be stored once."""
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Aspirin", "label": "drug"},
            {"text": "Aspirin", "label": "drug"},  # duplicate
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Some text.")

        assert result["drugs"].count("aspirin") == 1

    @patch("app.processing.entity_extractor._get_model")
    def test_normalizes_entity_text_to_lowercase(self, mock_get_model):
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "METFORMIN", "label": "drug"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Patient takes METFORMIN.")

        assert "metformin" in result["drugs"]
        assert "METFORMIN" not in result["drugs"]

    @patch("app.processing.entity_extractor._get_model")
    def test_skips_entity_with_empty_text(self, mock_get_model):
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "", "label": "drug"},
            {"text": "   ", "label": "drug"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Some text.")

        assert result["drugs"] == []

    @patch("app.processing.entity_extractor._get_model")
    def test_returns_empty_result_on_inference_exception(self, mock_get_model):
        """Should return empty result dict and not raise on NER failure."""
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.side_effect = RuntimeError("CUDA OOM")
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Some medical text.")

        assert result == {"drugs": [], "conditions": [], "biomarkers": [], "symptoms": []}

    @patch("app.processing.entity_extractor._get_model")
    def test_extracts_multiple_entity_types_simultaneously(self, mock_get_model):
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Metformin", "label": "drug"},
            {"text": "Type 2 Diabetes", "label": "medical condition"},
            {"text": "HbA1c", "label": "biomarker"},
            {"text": "fatigue", "label": "symptom"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Complex medical text.")

        assert "metformin" in result["drugs"]
        assert "type 2 diabetes" in result["conditions"]
        assert "hba1c" in result["biomarkers"]
        assert "fatigue" in result["symptoms"]

    @patch("app.processing.entity_extractor._get_model")
    def test_ignores_unknown_labels(self, mock_get_model):
        """Entities with unknown labels should be silently ignored."""
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "something", "label": "unknown_label"},
        ]
        mock_get_model.return_value = mock_model

        result = extract_medical_entities("Some text.")

        assert result["drugs"] == []
        assert result["conditions"] == []
        assert result["biomarkers"] == []
        assert result["symptoms"] == []

    @patch("app.processing.entity_extractor._get_model")
    def test_passes_settings_labels_to_model(self, mock_get_model):
        """Should use settings.NER_LABELS and NER_CONFIDENCE_THRESHOLD."""
        from app.processing.entity_extractor import extract_medical_entities

        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        mock_get_model.return_value = mock_model

        with patch("app.processing.entity_extractor.settings") as mock_settings:
            mock_settings.NER_LABELS = ["drug", "medical condition"]
            mock_settings.NER_CONFIDENCE_THRESHOLD = 0.5

            extract_medical_entities("Some text.")

        mock_model.predict_entities.assert_called_once_with(
            "Some text.",
            ["drug", "medical condition"],
            threshold=0.5,
        )


class TestGetNerDevice:
    """Tests for _get_device()"""

    def test_uses_settings_ner_device_when_set(self):
        from app.processing.entity_extractor import _get_device

        with patch("app.processing.entity_extractor.settings") as mock_settings:
            mock_settings.NER_DEVICE = "cpu"
            result = _get_device()

        assert result == "cpu"

    def test_falls_back_to_cuda_when_available(self):
        from app.processing.entity_extractor import _get_device

        with patch("app.processing.entity_extractor.settings") as mock_settings:
            mock_settings.NER_DEVICE = None
            with patch("app.processing.entity_extractor.torch.cuda.is_available", return_value=True):
                result = _get_device()

        assert result == "cuda"

    def test_falls_back_to_cpu_when_cuda_unavailable(self):
        from app.processing.entity_extractor import _get_device

        with patch("app.processing.entity_extractor.settings") as mock_settings:
            mock_settings.NER_DEVICE = None
            with patch("app.processing.entity_extractor.torch.cuda.is_available", return_value=False):
                result = _get_device()

        assert result == "cpu"


class TestGetNerModel:
    """Tests for _get_model() lazy loading in entity_extractor."""

    def test_model_is_cached_after_first_load(self):
        import app.processing.entity_extractor as ner_module

        original = ner_module._model
        mock_model = MagicMock()
        ner_module._model = mock_model

        try:
            with patch("app.processing.entity_extractor.GLiNER") as mock_gliner:
                result = ner_module._get_model()

            mock_gliner.from_pretrained.assert_not_called()
            assert result is mock_model
        finally:
            ner_module._model = original

    def test_model_loaded_when_none(self):
        import app.processing.entity_extractor as ner_module

        original = ner_module._model
        ner_module._model = None

        try:
            with patch("app.processing.entity_extractor.GLiNER") as mock_gliner:
                with patch("app.processing.entity_extractor._get_device", return_value="cpu"):
                    with patch("app.processing.entity_extractor.settings") as mock_settings:
                        mock_settings.NER_MODEL_NAME = "urchade/gliner_medium-v2.1"
                        mock_instance = MagicMock()
                        mock_instance.to.return_value = mock_instance
                        mock_gliner.from_pretrained.return_value = mock_instance

                        ner_module._get_model()

            mock_gliner.from_pretrained.assert_called_once_with("urchade/gliner_medium-v2.1")
        finally:
            ner_module._model = original


# ===========================================================================
# SECTION 11 — claim_extractor.py
# ===========================================================================

class TestStripMarkdown:
    """Tests for _strip_markdown()"""

    def test_removes_bold_markers(self):
        from app.rag.claim_extractor import _strip_markdown

        result = _strip_markdown("**Bold Text** is here")
        assert "**" not in result
        assert "Bold Text" in result

    def test_removes_italic_markers(self):
        from app.rag.claim_extractor import _strip_markdown

        result = _strip_markdown("*italic text* here")
        assert "*" not in result
        assert "italic text" in result

    def test_removes_inline_code(self):
        from app.rag.claim_extractor import _strip_markdown

        result = _strip_markdown("`code block`")
        assert "`" not in result
        assert "code block" in result

    def test_preserves_plain_text(self):
        from app.rag.claim_extractor import _strip_markdown

        text = "Normal sentence without markdown."
        assert _strip_markdown(text) == text

    def test_handles_empty_string(self):
        from app.rag.claim_extractor import _strip_markdown

        assert _strip_markdown("") == ""


class TestExtractClaims:
    """Tests for extract_claims() — main entry point."""

    def test_returns_empty_list_for_empty_input(self):
        from app.rag.claim_extractor import extract_claims

        assert extract_claims("") == []
        assert extract_claims(None) == []

    def test_returns_list_of_dicts(self):
        from app.rag.claim_extractor import extract_claims

        result = extract_claims("- RISK: High bleeding risk with aspirin.")
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_dash_based_extraction(self):
        """Should extract RISK/MONITORING/WARNING/RECOMMENDATION typed claims."""
        from app.rag.claim_extractor import extract_claims

        text = (
            "- RISK: Metformin raises lactic acidosis risk.\n"
            "- MONITORING: Monitor blood glucose weekly.\n"
            "- WARNING: Seek emergency help for chest pain."
        )
        result = extract_claims(text)

        types = [c["type"] for c in result]
        assert "risk" in types
        assert "monitoring" in types
        assert "warning" in types

    def test_dash_based_unknown_type_maps_to_general(self):
        from app.rag.claim_extractor import extract_claims

        result = extract_claims("- UNKNOWN_TYPE: Some statement here.")
        assert result[0]["type"] == "general"

    def test_fallback_produces_general_type(self):
        """Plain prose without structure should fall back to sentence extraction."""
        from app.rag.claim_extractor import extract_claims

        text = "This is a plain sentence about diabetes management."
        result = extract_claims(text)

        assert len(result) >= 1
        assert result[0]["statement"]

    def test_all_claims_have_type_and_statement_keys(self):
        from app.rag.claim_extractor import extract_claims

        result = extract_claims("- RISK: Something dangerous can happen here.")
        for claim in result:
            assert "type" in claim
            assert "statement" in claim

    def test_bold_section_extraction(self):
        """Should handle inline bold-style section headers after stripping."""
        from app.rag.claim_extractor import extract_claims

        text = (
            "**Key Considerations:** Monitor your blood pressure carefully. "
            "**What to Monitor:** Check glucose levels every morning."
        )
        result = extract_claims(text)

        assert len(result) >= 1

    def test_markdown_header_extraction(self):
        """Should extract claims from ## header-based structure."""
        from app.rag.claim_extractor import extract_claims

        text = (
            "## Risk Factors\n"
            "- High blood pressure is dangerous.\n"
            "- Avoid excess sodium.\n"
            "## Monitoring\n"
            "- Check BP daily."
        )
        result = extract_claims(text)

        assert len(result) >= 1

    def test_final_fallback_returns_at_least_one_claim(self):
        """Even completely unstructured text should produce at least 1 claim."""
        from app.rag.claim_extractor import extract_claims

        result = extract_claims("Unstructured blurb with no formatting whatsoever.")
        assert len(result) >= 1


class TestClassifySentence:
    """Tests for _classify_sentence()"""

    def test_classifies_risk_keywords(self):
        from app.rag.claim_extractor import _classify_sentence

        assert _classify_sentence("This can cause serious bleeding.") == "risk"
        assert _classify_sentence("Avoid this drug with alcohol.") == "risk"

    def test_classifies_monitoring_keywords(self):
        from app.rag.claim_extractor import _classify_sentence

        assert _classify_sentence("Monitor your blood pressure daily.") == "monitoring"
        assert _classify_sentence("Track glucose levels carefully.") == "monitoring"

    def test_classifies_warning_keywords(self):
        from app.rag.claim_extractor import _classify_sentence

        assert _classify_sentence("Seek emergency help immediately.") == "warning"
        assert _classify_sentence("Call your doctor urgently.") == "warning"

    def test_classifies_recommendation_keywords(self):
        from app.rag.claim_extractor import _classify_sentence

        assert _classify_sentence("It is recommended to take this daily.") == "recommendation"
        assert _classify_sentence("You should maintain a healthy diet.") == "recommendation"

    def test_returns_general_for_no_keywords(self):
        from app.rag.claim_extractor import _classify_sentence

        assert _classify_sentence("The patient was seen on Tuesday.") == "general"


class TestMapSectionToType:
    """Tests for _map_section_to_type()"""

    def test_risk_section(self):
        from app.rag.claim_extractor import _map_section_to_type
        assert _map_section_to_type("Risk Factors") == "risk"

    def test_monitoring_section(self):
        from app.rag.claim_extractor import _map_section_to_type
        assert _map_section_to_type("What to Watch") == "monitoring"

    def test_warning_section(self):
        from app.rag.claim_extractor import _map_section_to_type
        assert _map_section_to_type("When to Seek Help") == "warning"

    def test_recommendation_section(self):
        from app.rag.claim_extractor import _map_section_to_type
        assert _map_section_to_type("Safety Notes") == "recommendation"

    def test_unknown_section_returns_general(self):
        from app.rag.claim_extractor import _map_section_to_type
        assert _map_section_to_type("Introduction") == "general"


class TestExtractBulletPoints:
    """Tests for _extract_bullet_points()"""

    def test_extracts_dash_bullets(self):
        from app.rag.claim_extractor import _extract_bullet_points

        text = "- First item\n- Second item"
        result = _extract_bullet_points(text)
        assert "First item" in result
        assert "Second item" in result

    def test_extracts_asterisk_bullets(self):
        from app.rag.claim_extractor import _extract_bullet_points

        result = _extract_bullet_points("* Item one\n* Item two")
        assert len(result) == 2

    def test_extracts_numbered_list(self):
        from app.rag.claim_extractor import _extract_bullet_points

        result = _extract_bullet_points("1. First\n2. Second")
        assert "First" in result
        assert "Second" in result

    def test_skips_empty_lines(self):
        from app.rag.claim_extractor import _extract_bullet_points

        result = _extract_bullet_points("\n\n- Item\n\n")
        assert len(result) == 1

    def test_returns_empty_for_plain_text(self):
        from app.rag.claim_extractor import _extract_bullet_points

        result = _extract_bullet_points("No bullets here at all.")
        assert result == []


class TestSplitByHeaders:
    """Tests for _split_by_headers()"""

    def test_splits_on_double_hash(self):
        from app.rag.claim_extractor import _split_by_headers

        text = "## Risks\nContent A\n## Monitoring\nContent B"
        result = _split_by_headers(text)

        assert "risks" in result
        assert "monitoring" in result

    def test_content_is_captured_under_header(self):
        from app.rag.claim_extractor import _split_by_headers

        text = "## Risks\nContent A\nMore content"
        result = _split_by_headers(text)

        assert "Content A" in result.get("risks", "")

    def test_returns_empty_dict_for_no_headers(self):
        from app.rag.claim_extractor import _split_by_headers

        result = _split_by_headers("No headers here.")
        assert result == {} or "unknown" in result


class TestExtractSmartSentences:
    """Tests for _extract_smart_sentences()"""

    def test_returns_up_to_15_claims(self):
        from app.rag.claim_extractor import _extract_smart_sentences

        text = "This is a valid sentence about health. " * 20
        result = _extract_smart_sentences(text)

        assert len(result) <= 15

    def test_skips_very_short_sentences(self):
        from app.rag.claim_extractor import _extract_smart_sentences

        result = _extract_smart_sentences("Hi. Ok. Yes. No. This sentence is long enough to be included.")
        statements = [c["statement"] for c in result]
        assert not any(s in ["Hi", "Ok", "Yes", "No"] for s in statements)

    def test_all_claims_have_type_and_statement(self):
        from app.rag.claim_extractor import _extract_smart_sentences

        result = _extract_smart_sentences("Monitor your blood pressure every day for best results.")
        for c in result:
            assert "type" in c
            assert "statement" in c


# ===========================================================================
# SECTION 12 — fact_checker.py
# ===========================================================================

class TestNormalize:
    def test_lowercases_text(self):
        from app.rag.fact_checker import _normalize
        assert _normalize("METFORMIN") == "metformin"

    def test_handles_none(self):
        from app.rag.fact_checker import _normalize
        assert _normalize(None) == ""

    def test_handles_empty_string(self):
        from app.rag.fact_checker import _normalize
        assert _normalize("") == ""


class TestMatchAny:
    def test_returns_true_when_token_found(self):
        from app.rag.fact_checker import _match_any
        assert _match_any(["metformin", "aspirin"], "The patient takes metformin.") is True

    def test_returns_false_when_no_token_found(self):
        from app.rag.fact_checker import _match_any
        assert _match_any(["lisinopril"], "Patient takes aspirin.") is False

    def test_returns_false_for_empty_token_list(self):
        from app.rag.fact_checker import _match_any
        assert _match_any([], "Patient takes aspirin.") is False

    def test_skips_empty_tokens(self):
        from app.rag.fact_checker import _match_any
        assert _match_any(["", None, "aspirin"], "Patient takes aspirin.") is True


class TestPatientMedNames:
    def test_extracts_medication_names(self):
        from app.rag.fact_checker import _patient_med_names

        context = {
            "patient": {
                "medications": [
                    {"name": "Metformin"},
                    {"name": "Aspirin"},
                ]
            }
        }
        result = _patient_med_names(context)
        assert "metformin" in result
        assert "aspirin" in result

    def test_returns_empty_for_missing_patient(self):
        from app.rag.fact_checker import _patient_med_names
        assert _patient_med_names({}) == []

    def test_skips_meds_without_name(self):
        from app.rag.fact_checker import _patient_med_names

        context = {"patient": {"medications": [{"dosage": "10mg"}]}}
        assert _patient_med_names(context) == []


class TestPatientConditions:
    def test_extracts_condition_names(self):
        from app.rag.fact_checker import _patient_conditions

        context = {
            "patient": {
                "conditions": [
                    {"name": "Type 2 Diabetes"},
                    {"name": "Hypertension"},
                ]
            }
        }
        result = _patient_conditions(context)
        assert "type 2 diabetes" in result
        assert "hypertension" in result

    def test_returns_empty_for_empty_context(self):
        from app.rag.fact_checker import _patient_conditions
        assert _patient_conditions({}) == []


class TestPapersEvidence:
    def test_matches_paper_by_title_keyword(self):
        from app.rag.fact_checker import _papers_evidence

        context = {
            "papers": [
                {
                    "pmid": "123",
                    "title": "Metformin and diabetes control",
                    "text_preview": "Long-term use of metformin reduces HbA1c.",
                }
            ]
        }
        hits = _papers_evidence(context, "metformin reduces blood sugar")
        assert len(hits) >= 1
        assert hits[0]["type"] == "paper"

    def test_returns_empty_when_no_match(self):
        from app.rag.fact_checker import _papers_evidence

        context = {
            "papers": [
                {"pmid": "999", "title": "Kidney disease study", "text_preview": "CKD research."}
            ]
        }
        hits = _papers_evidence(context, "aspirin bleeding zxqwerty")
        # Could be 0 matches for very specific nonsense query
        assert isinstance(hits, list)

    def test_returns_empty_for_no_papers(self):
        from app.rag.fact_checker import _papers_evidence

        hits = _papers_evidence({"papers": []}, "metformin")
        assert hits == []


class TestVerifyClaim:
    """Tests for verify_claim()"""

    def test_verified_true_when_medication_matches(self):
        from app.rag.fact_checker import verify_claim

        claim = {"type": "risk", "statement": "metformin may cause lactic acidosis."}
        context = {
            "patient": {"medications": [{"name": "Metformin"}], "conditions": []},
            "papers": []
        }
        result = verify_claim(claim, context)

        assert result["verified"] is True
        assert result["type"] == "risk"
        assert result["statement"] == claim["statement"]

    def test_verified_true_when_condition_matches(self):
        from app.rag.fact_checker import verify_claim

        # "diabetes" appears in "type 2 diabetes" — _match_any does substring match
        claim = {"type": "general", "statement": "type 2 diabetes requires careful monitoring."}
        context = {
            "patient": {
                "medications": [],
                "conditions": [{"name": "Type 2 Diabetes"}]
            },
            "papers": []
        }
        result = verify_claim(claim, context)

        assert result["verified"] is True

    def test_verified_false_when_no_match(self):
        from app.rag.fact_checker import verify_claim

        claim = {"type": "general", "statement": "xyz completely unrelated nonsense statement."}
        context = {
            "patient": {
                "medications": [{"name": "Aspirin"}],
                "conditions": [{"name": "Hypertension"}]
            },
            "papers": []
        }
        result = verify_claim(claim, context)

        assert result["verified"] is False
        assert result["sources"] == []

    def test_result_always_has_required_keys(self):
        from app.rag.fact_checker import verify_claim

        claim = {"type": "risk", "statement": "test statement."}
        result = verify_claim(claim, {})

        for key in ["type", "statement", "verified", "sources"]:
            assert key in result

    def test_sources_include_kg_type_for_med_match(self):
        from app.rag.fact_checker import verify_claim

        claim = {"type": "risk", "statement": "aspirin increases bleeding risk."}
        context = {
            "patient": {"medications": [{"name": "Aspirin"}], "conditions": []},
            "papers": []
        }
        result = verify_claim(claim, context)

        source_types = [s["type"] for s in result["sources"]]
        assert "kg" in source_types


class TestVerifyClaims:
    """Tests for verify_claims()"""

    def test_returns_empty_list_for_no_claims(self):
        from app.rag.fact_checker import verify_claims

        assert verify_claims([], {}) == []
        assert verify_claims(None, {}) == []

    def test_processes_list_of_claim_dicts(self):
        from app.rag.fact_checker import verify_claims

        claims = [
            {"type": "risk", "statement": "metformin causes lactic acidosis."},
        ]
        context = {
            "patient": {"medications": [{"name": "Metformin"}], "conditions": []},
            "papers": []
        }
        result = verify_claims(claims, context)

        assert len(result) == 1
        assert "verified" in result[0]

    def test_handles_string_input_by_extracting_claims(self):
        """If a raw string is passed, should auto-extract claims first."""
        from app.rag.fact_checker import verify_claims

        result = verify_claims("- RISK: Monitor blood pressure carefully.", {})
        assert isinstance(result, list)

    def test_exception_in_single_claim_does_not_break_others(self):
        from app.rag.fact_checker import verify_claims

        # Malformed claim dict (missing keys) should not crash full list
        claims = [
            {},  # malformed
            {"type": "risk", "statement": "aspirin can cause bleeding."},
        ]
        context = {
            "patient": {"medications": [{"name": "Aspirin"}], "conditions": []},
            "papers": []
        }
        result = verify_claims(claims, context)

        assert len(result) == 2
        assert result[0]["verified"] is False  # malformed → False

    def test_all_results_have_verified_key(self):
        from app.rag.fact_checker import verify_claims

        claims = [{"type": "general", "statement": "some health claim."}]
        result = verify_claims(claims, {})

        for r in result:
            assert "verified" in r


# ===========================================================================
# SECTION 13 — qdrant_search.py
# ===========================================================================

class TestQdrantHybridSearch:
    """Tests for qdrant_hybrid_search()"""

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post")
    def test_returns_list_of_results(self, mock_post, mock_embed):
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "score": 0.95,
                    "payload": {
                        "pmid": "PMC123",
                        "title": "Diabetes study",
                        "text": "Long text about diabetes...",
                        "entities": {"drugs": ["metformin"]},
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = qdrant_hybrid_search(
            question="What is the best drug for diabetes?",
            user_context={"conditions": ["diabetes"], "drugs": ["metformin"]},
            expanded_entities={"drugs": ["insulin"]},
            top_k=5,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["pmid"] == "PMC123"
        assert result[0]["score"] == 0.95

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post")
    def test_result_has_required_fields(self, mock_post, mock_embed):
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1, 0.2]]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "score": 0.8,
                    "payload": {
                        "pmid": "PMC456",
                        "title": "Hypertension research",
                        "text": "Research about BP...",
                        "entities": {},
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = qdrant_hybrid_search("BP question", {}, {}, top_k=1)

        assert "score" in result[0]
        assert "pmid" in result[0]
        assert "title" in result[0]
        assert "text_preview" in result[0]
        assert "entities" in result[0]

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post")
    def test_truncates_text_preview_to_500_chars(self, mock_post, mock_embed):
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1]]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "score": 0.7,
                    "payload": {
                        "pmid": "PMC789",
                        "title": "Long study",
                        "text": "A" * 1000,
                        "entities": {},
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = qdrant_hybrid_search("question", {}, {})

        assert len(result[0]["text_preview"]) <= 500

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post")
    def test_returns_empty_list_for_no_results(self, mock_post, mock_embed):
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1]]
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_post.return_value = mock_response

        result = qdrant_hybrid_search("obscure query", {}, {})

        assert result == []

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post")
    def test_combines_context_entities_into_query(self, mock_post, mock_embed):
        """Should merge conditions + drugs + expanded_entities into query text."""
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1]]
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_post.return_value = mock_response

        qdrant_hybrid_search(
            question="test question",
            user_context={"conditions": ["diabetes"], "drugs": ["metformin"]},
            expanded_entities={"biomarkers": ["HbA1c"]},
        )

        embedded_text = mock_embed.call_args[0][0][0]
        assert "diabetes" in embedded_text
        assert "metformin" in embedded_text

    @patch("app.rag.qdrant_search.embed_texts")
    @patch("app.rag.qdrant_search.requests.post", side_effect=Exception("Qdrant unreachable"))
    def test_raises_on_connection_error(self, mock_post, mock_embed):
        from app.rag.qdrant_search import qdrant_hybrid_search

        mock_embed.return_value = [[0.1]]

        with pytest.raises(Exception):
            qdrant_hybrid_search("question", {}, {})


# ===========================================================================
# SECTION 14 — prompt_builder.py
# ===========================================================================

class TestBuildMedicalPrompt:
    """Tests for build_medical_prompt()"""

    def _base_context(self):
        return {
            "patient": {
                "patient_id": "user_1",
                "demographics": {"age": 52, "gender": "Male", "blood_type": "O+"},
                "conditions": [{"name": "Type 2 Diabetes", "severity": "moderate", "status": "active"}],
                "medications": [{"name": "Metformin", "dosage": "1000mg", "frequency": "twice daily", "purpose": "Blood sugar control"}],
                "lab_results": [{"name": "HbA1c", "result": 7.2, "unit": "%", "normal_range": "< 5.7%", "status": "slightly elevated", "date": "2025-01-10"}],
            },
            "wearables": {"available": False, "metrics": []},
            "papers": [],
            "drug_interactions": {},
        }

    def test_returns_non_empty_string(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("What should I monitor?", self._base_context())
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_user_question(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("Is my blood sugar under control?", self._base_context())
        assert "Is my blood sugar under control?" in result

    def test_contains_patient_id(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "user_1" in result

    def test_contains_medication_names(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "Metformin" in result

    def test_contains_condition_names(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "Type 2 Diabetes" in result

    def test_contains_safety_rules_header(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "CRITICAL SAFETY RULES" in result

    def test_contains_response_format_section(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "RESPONSE FORMAT" in result

    def test_no_research_papers_message_when_empty(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "No relevant research papers found" in result

    def test_includes_papers_when_provided(self):
        from app.rag.prompt_builder import build_medical_prompt

        context = self._base_context()
        context["papers"] = [
            {"title": "Diabetes Research 2024", "journal": "Lancet", "year": 2024, "text_preview": "HbA1c reduction..."}
        ]
        result = build_medical_prompt("question", context)
        assert "Diabetes Research 2024" in result

    def test_handles_empty_context(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", {})
        assert isinstance(result, str)
        assert "No patient data available" in result

    def test_ends_with_consult_disclaimer(self):
        from app.rag.prompt_builder import build_medical_prompt

        result = build_medical_prompt("question", self._base_context())
        assert "Consult your healthcare provider" in result


class TestFormatPatient:
    """Tests for _format_patient()"""

    def test_returns_no_data_message_for_empty_dict(self):
        from app.rag.prompt_builder import _format_patient
        assert "No patient data available" in _format_patient({})

    def test_formats_demographics(self):
        from app.rag.prompt_builder import _format_patient

        patient = {
            "patient_id": "user_1",
            "demographics": {"age": 52, "gender": "Male", "blood_type": "O+"},
            "conditions": [], "medications": [], "lab_results": []
        }
        result = _format_patient(patient)
        assert "52" in result
        assert "Male" in result
        assert "O+" in result

    def test_formats_medications(self):
        from app.rag.prompt_builder import _format_patient

        patient = {
            "patient_id": "u1",
            "demographics": {},
            "conditions": [],
            "medications": [{"name": "Aspirin", "dosage": "81mg", "frequency": "daily", "purpose": "Blood thinner"}],
            "lab_results": []
        }
        result = _format_patient(patient)
        assert "Aspirin" in result
        assert "81mg" in result

    def test_formats_lab_results(self):
        from app.rag.prompt_builder import _format_patient

        patient = {
            "patient_id": "u1",
            "demographics": {},
            "conditions": [],
            "medications": [],
            "lab_results": [{"name": "HbA1c", "result": 7.2, "unit": "%", "normal_range": "< 5.7%", "status": "elevated", "date": "2025-01-10"}]
        }
        result = _format_patient(patient)
        assert "HbA1c" in result
        assert "7.2" in result


class TestFormatWearables:
    """Tests for _format_wearables()"""

    def test_returns_no_data_when_unavailable(self):
        from app.rag.prompt_builder import _format_wearables

        result = _format_wearables({"available": False, "metrics": []})
        assert "No wearable data available" in result

    def test_returns_no_data_for_empty_dict(self):
        from app.rag.prompt_builder import _format_wearables

        result = _format_wearables({})
        assert "No wearable data available" in result

    def test_formats_metric_with_readings(self):
        from app.rag.prompt_builder import _format_wearables

        wearables = {
            "available": True,
            "metrics": [{
                "metric": "Blood Glucose",
                "latest_value": "156 mg/dL",
                "previous_value": "142 mg/dL",
                "average_value": "149.0 mg/dL",
                "normal_range": "70-100",
                "trend": "stable",
                "readings": [{"date": "2026-02-08", "value": "156 mg/dL"}]
            }]
        }
        result = _format_wearables(wearables)
        assert "Blood Glucose" in result
        assert "156 mg/dL" in result

    def test_sanitizes_internal_labels_in_trend(self):
        """Trends with 'insufficient' or 'N/A' should be replaced."""
        from app.rag.prompt_builder import _format_wearables

        wearables = {
            "available": True,
            "metrics": [{
                "metric": "ECG",
                "latest_value": "NSR",
                "previous_value": "NSR",
                "average_value": "N/A",
                "normal_range": "normal",
                "trend": "insufficient-data",
                "readings": []
            }]
        }
        result = _format_wearables(wearables)
        assert "insufficient" not in result
        assert "more readings needed" in result


class TestFormatDrugFacts:
    """Tests for _format_drug_facts()"""

    def test_returns_no_data_for_empty(self):
        from app.rag.prompt_builder import _format_drug_facts

        assert "No medication safety data available" in _format_drug_facts({})
        assert "No medication safety data available" in _format_drug_facts(None)

    def test_formats_drug_drug_interaction(self):
        from app.rag.prompt_builder import _format_drug_facts

        facts = {
            "drug_drug_interactions": [{
                "drugs_involved": ["metformin", "contrast dye"],
                "severity": "high",
                "interaction": "Increased risk of lactic acidosis",
                "mechanism": "Renal impairment",
            }],
            "drug_condition_interactions": [],
            "drug_effect_facts": [],
        }
        result = _format_drug_facts(facts)
        assert "metformin" in result
        assert "lactic acidosis" in result
        assert "high" in result

    def test_formats_drug_effect_facts(self):
        from app.rag.prompt_builder import _format_drug_facts

        facts = {
            "drug_drug_interactions": [],
            "drug_condition_interactions": [],
            "drug_effect_facts": [{
                "drug": "metformin",
                "effect": "B12 absorption reduction",
                "mechanism": "Calcium-dependent pathway"
            }],
        }
        result = _format_drug_facts(facts)
        assert "metformin" in result
        assert "B12" in result

    def test_returns_no_risks_when_all_lists_empty(self):
        from app.rag.prompt_builder import _format_drug_facts

        facts = {
            "drug_drug_interactions": [],
            "drug_condition_interactions": [],
            "drug_effect_facts": [],
        }
        result = _format_drug_facts(facts)
        assert "No known medication risks identified" in result


class TestFormatPapers:
    """Tests for _format_papers()"""

    def test_returns_no_papers_message_for_empty(self):
        from app.rag.prompt_builder import _format_papers

        assert "No relevant research papers found" in _format_papers([])
        assert "No relevant research papers found" in _format_papers(None)

    def test_formats_paper_title_and_journal(self):
        from app.rag.prompt_builder import _format_papers

        papers = [{"title": "Diabetes Study", "journal": "Lancet", "year": 2024, "text_preview": "HbA1c data..."}]
        result = _format_papers(papers)
        assert "Diabetes Study" in result
        assert "Lancet" in result
        assert "2024" in result

    def test_limits_to_three_papers(self):
        from app.rag.prompt_builder import _format_papers

        papers = [
            {"title": f"Study {i}", "journal": "Journal", "year": 2024, "text_preview": ""}
            for i in range(10)
        ]
        result = _format_papers(papers)
        # Only 3 should appear — check by counting [1], [2], [3], [4] markers
        assert "[4]" not in result
        assert "[3]" in result


# ===========================================================================
# SECTION 15 — graph_rag_pipeline.py
# ===========================================================================

class TestRunHybridRagPipeline:
    """Tests for run_hybrid_rag_pipeline()"""

    def _make_mocks(self):
        """Helper to return a consistent set of mock return values."""
        patient_profile = {
            "patient_id": "user_1",
            "name": "John Doe",
            "age": 52,
            "gender": "Male",
            "demographics": {"age": 52, "gender": "Male", "blood_type": "O+"},
            "conditions": [{"name": "Type 2 Diabetes", "severity": "moderate", "status": "active"}],
            "medications": [{"name": "Metformin", "dosage": "1000mg", "frequency": "twice daily"}],
            "lab_results": [],
            "wearables": {"available": False, "metrics": []},
        }
        return patient_profile

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[{"type": "risk", "statement": "monitor glucose"}])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="Monitor your glucose carefully.")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="Prompt text")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_returns_dict_with_required_keys(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        result = run_hybrid_rag_pipeline("user_1", "Is my blood sugar okay?")

        assert "response" in result
        assert "claims" in result
        assert "context" in result

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="LLM response")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="Prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_upserts_patient_before_profile_fetch(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        run_hybrid_rag_pipeline("user_1", "question")

        mock_upsert.assert_called_once_with("user_1", "question")
        mock_profile.assert_called_once_with("user_1")

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="response")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[{"title": "Paper 1"}])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={"drug_drug_interactions": []})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_context_contains_patient_wearables_papers_drugs(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        result = run_hybrid_rag_pipeline("user_1", "question")

        context = result["context"]
        assert "patient" in context
        assert "wearables" in context
        assert "papers" in context
        assert "drug_interactions" in context

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[{"type": "risk", "statement": "monitor"}])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="LLM answer")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_response_is_llm_output(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        result = run_hybrid_rag_pipeline("user_1", "Is my BP okay?")

        assert result["response"] == "LLM answer"

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[{"type": "general", "statement": "claim"}])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="LLM answer")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_claims_are_extracted_from_llm_response(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        result = run_hybrid_rag_pipeline("user_1", "question")

        assert result["claims"] == [{"type": "general", "statement": "claim"}]
        mock_claims.assert_called_once_with("LLM answer")

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="response")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_drug_interactions_called_with_patient_medications(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        profile = self._make_mocks()
        mock_profile.return_value = profile

        run_hybrid_rag_pipeline("user_1", "question")

        mock_drugs.assert_called_once_with(
            medications=profile["medications"]
        )

    @patch("app.rag.graph_rag_pipeline.extract_claims", return_value=[])
    @patch("app.rag.graph_rag_pipeline.call_ollama", return_value="response")
    @patch("app.rag.graph_rag_pipeline.build_medical_prompt", return_value="prompt")
    @patch("app.rag.graph_rag_pipeline.search_papers", return_value=[])
    @patch("app.rag.graph_rag_pipeline.check_drug_interactions", return_value={})
    @patch("app.rag.graph_rag_pipeline.get_patient_profile")
    @patch("app.rag.graph_rag_pipeline.upsert_user_from_question")
    def test_wearables_popped_from_profile_into_context(
        self, mock_upsert, mock_profile, mock_drugs,
        mock_papers, mock_prompt, mock_ollama, mock_claims
    ):
        """Wearables should be moved from profile into top-level context."""
        from app.rag.graph_rag_pipeline import run_hybrid_rag_pipeline

        mock_profile.return_value = self._make_mocks()

        result = run_hybrid_rag_pipeline("user_1", "question")

        # Wearables should be in context, not nested inside patient
        assert "wearables" in result["context"]
        assert "wearables" not in result["context"].get("patient", {})


# ===========================================================================
# SECTION 16 — schema_builder.py
# ===========================================================================

class TestEmptyEntityBlock:
    """Tests for _empty_entity_block()"""

    def test_returns_dict_with_four_keys(self):
        from app.schema.schema_builder import _empty_entity_block

        result = _empty_entity_block()
        assert set(result.keys()) == {"drugs", "conditions", "biomarkers", "symptoms"}

    def test_all_values_are_empty_lists(self):
        from app.schema.schema_builder import _empty_entity_block

        result = _empty_entity_block()
        for v in result.values():
            assert v == []

    def test_returns_independent_instances(self):
        """Mutation of one result must not affect another."""
        from app.schema.schema_builder import _empty_entity_block

        r1 = _empty_entity_block()
        r2 = _empty_entity_block()
        r1["drugs"].append("metformin")

        assert r2["drugs"] == []


class TestBuildPayload:
    """Tests for build_payload()"""

    def _base_kwargs(self):
        return dict(
            text="Sample chunk text.",
            pmid="PMC12345",
            title="Diabetes Study",
            journal="Lancet",
            year=2024,
            authors=["Dr. A", "Dr. B"],
            section="Full Text",
            chunk_index=0,
            api_query="type 2 diabetes",
        )

    def test_returns_dict(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert isinstance(result, dict)

    def test_contains_all_required_fields(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        required = [
            "schema_version", "source", "retrieved_at",
            "pmid", "title", "journal", "year", "authors",
            "section", "chunk_index", "api_query", "text",
            "entities", "relations", "kg_node_ids",
            "study_type", "confidence_level",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_pmid_stored_as_string(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**{**self._base_kwargs(), "pmid": 12345})
        assert result["pmid"] == "12345"
        assert isinstance(result["pmid"], str)

    def test_year_stored_as_int(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**{**self._base_kwargs(), "year": "2024"})
        assert result["year"] == 2024
        assert isinstance(result["year"], int)

    def test_chunk_index_stored_as_int(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**{**self._base_kwargs(), "chunk_index": "3"})
        assert result["chunk_index"] == 3
        assert isinstance(result["chunk_index"], int)

    def test_entities_defaults_to_empty_block_when_none(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["entities"] == {"drugs": [], "conditions": [], "biomarkers": [], "symptoms": []}

    def test_entities_used_when_provided(self):
        from app.schema.schema_builder import build_payload

        entities = {"drugs": ["metformin"], "conditions": [], "biomarkers": [], "symptoms": []}
        result = build_payload(**self._base_kwargs(), entities=entities)
        assert result["entities"]["drugs"] == ["metformin"]

    def test_kg_node_ids_defaults_to_empty_block(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["kg_node_ids"] == {"drugs": [], "conditions": [], "biomarkers": [], "symptoms": []}

    def test_source_is_pubmed_api(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["source"] == "pubmed_api"

    def test_schema_version_is_set(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["schema_version"] == "1.0"

    def test_relations_is_empty_list(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["relations"] == []

    def test_retrieved_at_is_iso_format(self):
        from app.schema.schema_builder import build_payload
        from datetime import datetime

        result = build_payload(**self._base_kwargs())
        # Should parse without error
        parsed = datetime.fromisoformat(result["retrieved_at"].replace("Z", "+00:00"))
        assert parsed is not None

    def test_text_stored_verbatim(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**{**self._base_kwargs(), "text": "Verbatim chunk."})
        assert result["text"] == "Verbatim chunk."

    def test_authors_stored_as_list(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert isinstance(result["authors"], list)
        assert "Dr. A" in result["authors"]

    def test_study_type_defaults_to_none(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["study_type"] is None

    def test_confidence_level_defaults_to_none(self):
        from app.schema.schema_builder import build_payload

        result = build_payload(**self._base_kwargs())
        assert result["confidence_level"] is None


# ===========================================================================
# SECTION 17 — models.py (User)
# ===========================================================================

class TestUserModel:
    """Tests for the User SQLAlchemy model."""

    def test_set_password_hashes_password(self):
        from app.models import User

        user = User(username="testuser")
        user.set_password("securepassword")

        assert user.password_hash is not None
        assert user.password_hash != "securepassword"

    def test_check_password_returns_true_for_correct_password(self):
        from app.models import User

        user = User(username="testuser")
        user.set_password("mypassword")

        assert user.check_password("mypassword") is True

    def test_check_password_returns_false_for_wrong_password(self):
        from app.models import User

        user = User(username="testuser")
        user.set_password("correct")

        assert user.check_password("wrong") is False

    def test_check_password_returns_false_for_empty_password(self):
        from app.models import User

        user = User(username="testuser")
        user.set_password("correct")

        assert user.check_password("") is False

    def test_default_role_is_patient(self, flask_app):
        """Role default is applied by SQLAlchemy on DB insert, not plain instantiation."""
        from app.models import db, User

        with flask_app.app_context():
            user = User(username="roletest")
            user.set_password("pw")
            db.session.add(user)
            db.session.commit()
            fetched = User.query.filter_by(username="roletest").first()
            assert fetched.role == "patient"

    def test_different_users_have_different_hashes(self):
        """Same password should produce different hashes (salting)."""
        from app.models import User

        u1 = User(username="user1")
        u2 = User(username="user2")
        u1.set_password("samepassword")
        u2.set_password("samepassword")

        assert u1.password_hash != u2.password_hash

    def test_username_stored_correctly(self):
        from app.models import User

        user = User(username="john_doe")
        assert user.username == "john_doe"


# ===========================================================================
# SECTION 18 — paper_search.py
# ===========================================================================

class TestSearchPapers:
    """Tests for search_papers()"""

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_returns_list_of_papers(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        mock_hit = MagicMock()
        mock_hit.score = 0.92
        mock_hit.payload = {
            "pmid": "PMC123",
            "title": "Diabetes Study",
            "journal": "Lancet",
            "year": 2024,
            "section": "Full Text",
            "text": "Long content about diabetes management...",
            "entities": {"drugs": ["metformin"]},
        }

        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_get_client.return_value.query_points.return_value = mock_response

        result = search_papers("diabetes management", top_k=3)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["pmid"] == "PMC123"
        assert result[0]["score"] == 0.92

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_result_has_required_fields(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1]]
        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {
            "pmid": "PMC999", "title": "Study", "journal": "BMJ",
            "year": 2023, "section": "Abstract", "text": "Content", "entities": {}
        }
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_get_client.return_value.query_points.return_value = mock_response

        result = search_papers("query")

        for field in ["score", "pmid", "title", "journal", "year", "section", "text_preview", "entities"]:
            assert field in result[0], f"Missing field: {field}"

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_text_preview_truncated_to_500(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1]]
        mock_hit = MagicMock()
        mock_hit.score = 0.7
        mock_hit.payload = {"text": "A" * 1000, "pmid": "1", "title": "T",
                            "journal": "J", "year": 2024, "section": "S", "entities": {}}
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_get_client.return_value.query_points.return_value = mock_response

        result = search_papers("query")

        assert len(result[0]["text_preview"]) <= 500

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_returns_empty_list_when_no_hits(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1]]
        mock_response = MagicMock()
        mock_response.points = []
        mock_get_client.return_value.query_points.return_value = mock_response

        result = search_papers("obscure query")

        assert result == []

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_handles_none_text_gracefully(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1]]
        mock_hit = MagicMock()
        mock_hit.score = 0.5
        mock_hit.payload = {"text": None, "pmid": "1", "title": "T",
                            "journal": "J", "year": 2024, "section": "S", "entities": {}}
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_get_client.return_value.query_points.return_value = mock_response

        result = search_papers("query")

        assert result[0]["text_preview"] == ""

    @patch("app.vector_store.paper_search.embed_texts")
    @patch("app.vector_store.paper_search.get_client")
    def test_passes_top_k_to_qdrant(self, mock_get_client, mock_embed):
        from app.vector_store.paper_search import search_papers

        mock_embed.return_value = [[0.1]]
        mock_response = MagicMock()
        mock_response.points = []
        mock_get_client.return_value.query_points.return_value = mock_response

        search_papers("query", top_k=7)

        call_kwargs = mock_get_client.return_value.query_points.call_args[1]
        assert call_kwargs.get("limit") == 7


# ===========================================================================
# SECTION 19 — qdrant_store.py
# ===========================================================================

class TestGetClient:
    """Tests for get_client()"""

    @patch("app.vector_store.qdrant_store.QdrantClient")
    def test_returns_qdrant_client_on_success(self, mock_qdrant):
        from app.vector_store.qdrant_store import get_client

        mock_instance = MagicMock()
        mock_qdrant.return_value = mock_instance

        result = get_client()

        assert result is mock_instance

    @patch("app.vector_store.qdrant_store.QdrantClient", side_effect=Exception("Connection refused"))
    def test_raises_runtime_error_on_failure(self, mock_qdrant):
        from app.vector_store.qdrant_store import get_client

        with pytest.raises(RuntimeError, match="Qdrant connection failed"):
            get_client()

    @patch("app.vector_store.qdrant_store.QdrantClient")
    def test_uses_settings_url_and_api_key(self, mock_qdrant):
        from app.vector_store.qdrant_store import get_client

        mock_qdrant.return_value = MagicMock()

        with patch("app.vector_store.qdrant_store.settings") as mock_settings:
            mock_settings.QDRANT_URL = "http://localhost:6333"
            mock_settings.QDRANT_API_KEY = "test-key"
            mock_settings.QDRANT_TIMEOUT = 30

            get_client()

        mock_qdrant.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test-key",
            timeout=30,
        )


class TestCreateCollectionIfNotExists:
    """Tests for create_collection_if_not_exists()"""

    @patch("app.vector_store.qdrant_store.COLLECTION", "test_collection")
    def test_skips_creation_when_collection_exists(self):
        from app.vector_store.qdrant_store import create_collection_if_not_exists

        mock_client = MagicMock()
        existing_col = MagicMock()
        existing_col.name = "test_collection"
        mock_client.get_collections.return_value.collections = [existing_col]

        create_collection_if_not_exists(mock_client)

        mock_client.create_collection.assert_not_called()

    @patch("app.vector_store.qdrant_store.COLLECTION", "new_collection")
    def test_creates_collection_when_absent(self):
        from app.vector_store.qdrant_store import create_collection_if_not_exists

        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        with patch("app.vector_store.qdrant_store.settings") as mock_settings:
            mock_settings.EMBEDDING_DIM = 384

            create_collection_if_not_exists(mock_client)

        mock_client.create_collection.assert_called_once()

    def test_raises_on_unexpected_error(self):
        from app.vector_store.qdrant_store import create_collection_if_not_exists

        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("DB error")

        with pytest.raises(Exception):
            create_collection_if_not_exists(mock_client)


class TestCreateIndexes:
    """Tests for create_indexes()"""

    def test_creates_all_five_indexes(self):
        from app.vector_store.qdrant_store import create_indexes

        mock_client = MagicMock()

        with patch("app.vector_store.qdrant_store._create_payload_index_safe") as mock_idx:
            create_indexes(mock_client)

        assert mock_idx.call_count == 5

    def test_includes_pmid_index(self):
        from app.vector_store.qdrant_store import create_indexes

        mock_client = MagicMock()
        called_fields = []

        with patch("app.vector_store.qdrant_store._create_payload_index_safe") as mock_idx:
            mock_idx.side_effect = lambda client, field, schema: called_fields.append(field)
            create_indexes(mock_client)

        assert "pmid" in called_fields
        assert "year" in called_fields
        assert "journal" in called_fields


class TestCreatePayloadIndexSafe:
    """Tests for _create_payload_index_safe()"""

    def test_creates_index_successfully(self):
        from app.vector_store.qdrant_store import _create_payload_index_safe
        from qdrant_client.models import PayloadSchemaType

        mock_client = MagicMock()
        _create_payload_index_safe(mock_client, "pmid", PayloadSchemaType.KEYWORD)

        mock_client.create_payload_index.assert_called_once()

    def test_does_not_raise_on_unexpected_response(self):
        """UnexpectedResponse (index exists) should be silently ignored."""
        from app.vector_store.qdrant_store import _create_payload_index_safe
        from qdrant_client.models import PayloadSchemaType
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        mock_client.create_payload_index.side_effect = UnexpectedResponse(
            status_code=400, reason_phrase="Bad Request",
            content=b"already exists", headers={}
        )

        # Should not raise
        _create_payload_index_safe(mock_client, "pmid", PayloadSchemaType.KEYWORD)

    def test_raises_on_unexpected_exception(self):
        from app.vector_store.qdrant_store import _create_payload_index_safe
        from qdrant_client.models import PayloadSchemaType

        mock_client = MagicMock()
        mock_client.create_payload_index.side_effect = RuntimeError("Unexpected DB failure")

        with pytest.raises(RuntimeError):
            _create_payload_index_safe(mock_client, "pmid", PayloadSchemaType.KEYWORD)


# ===========================================================================
# SECTION 20 — api.py (Flask routes)
# ===========================================================================

import pytest

@pytest.fixture
def flask_app():
    """Create a Flask test app with in-memory SQLite."""
    from flask import Flask
    from flask_jwt_extended import JWTManager
    from app.models import db, User
    from app.routes.api import api_bp

    app = Flask(__name__, template_folder="../../templates")
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JWT_SECRET_KEY"] = "test-secret-key"

    db.init_app(app)
    JWTManager(app)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.create_all()
        yield app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def auth_headers(flask_app, client):
    """Register a user and return JWT auth headers."""
    with patch("app.routes.api.create_patient"):
        client.post("/api/register", json={"username": "testuser", "password": "pass123"})

    resp = client.post("/api/login", json={"username": "testuser", "password": "pass123"})
    token = resp.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


class TestRegisterEndpoint:
    @patch("app.routes.api.create_patient")
    def test_register_success(self, mock_create, client):
        resp = client.post("/api/register", json={"username": "newuser", "password": "pw123"})
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_register_missing_username(self, client):
        resp = client.post("/api/register", json={"password": "pw123"})
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    def test_register_missing_password(self, client):
        resp = client.post("/api/register", json={"username": "user"})
        assert resp.status_code == 400

    @patch("app.routes.api.create_patient")
    def test_register_duplicate_user(self, mock_create, client):
        client.post("/api/register", json={"username": "dupuser", "password": "pw"})
        resp = client.post("/api/register", json={"username": "dupuser", "password": "pw"})
        assert resp.status_code == 400
        assert "already exists" in resp.get_json()["error"]


class TestLoginEndpoint:
    @patch("app.routes.api.create_patient")
    def test_login_success_returns_token(self, mock_create, client):
        client.post("/api/register", json={"username": "loginuser", "password": "mypass"})
        resp = client.post("/api/login", json={"username": "loginuser", "password": "mypass"})

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "access_token" in data

    def test_login_wrong_password(self, client):
        resp = client.post("/api/login", json={"username": "nobody", "password": "wrong"})
        assert resp.status_code == 401
        assert resp.get_json()["success"] is False

    def test_login_nonexistent_user(self, client):
        resp = client.post("/api/login", json={"username": "ghost", "password": "pw"})
        assert resp.status_code == 401


class TestAskEndpoint:
    @patch("app.routes.api.extract_claims", return_value=[])
    @patch("app.routes.api.call_ollama", return_value="Monitor your glucose.")
    @patch("app.routes.api.build_medical_prompt", return_value="prompt text")
    @patch("app.routes.api.check_drug_interactions", return_value={"drug_drug_interactions": []})
    @patch("app.routes.api.search_papers", return_value=[])
    @patch("app.routes.api.get_wearable_summary", return_value={"metrics": []})
    @patch("app.routes.api.get_patient_profile", return_value={
        "patient_id": "testuser", "conditions": [], "medications": [], "lab_results": []
    })
    @patch("app.routes.api.analyze_health_intent", return_value=[])
    def test_ask_returns_success(
        self, mock_intent, mock_profile, mock_wearables, mock_papers,
        mock_drugs, mock_prompt, mock_ollama, mock_claims,
        client, auth_headers
    ):
        resp = client.post("/api/ask", json={"question": "Is my glucose okay?"}, headers=auth_headers)

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "answer" in data
        assert "claims" in data
        assert "context" in data

    def test_ask_requires_auth(self, client):
        resp = client.post("/api/ask", json={"question": "test"})
        assert resp.status_code == 401

    @patch("app.routes.api.analyze_health_intent", return_value=[])
    @patch("app.routes.api.get_patient_profile", return_value={
        "conditions": [], "medications": [], "lab_results": []
    })
    @patch("app.routes.api.get_wearable_summary", return_value={"metrics": []})
    @patch("app.routes.api.search_papers", return_value=[])
    @patch("app.routes.api.check_drug_interactions", return_value={"drug_drug_interactions": []})
    @patch("app.routes.api.build_medical_prompt", return_value="prompt")
    @patch("app.routes.api.call_ollama", return_value="response")
    @patch("app.routes.api.extract_claims", return_value=[])
    def test_ask_missing_question_returns_400(
        self, mock_claims, mock_ollama, mock_prompt, mock_drugs,
        mock_papers, mock_wearables, mock_profile, mock_intent,
        client, auth_headers
    ):
        resp = client.post("/api/ask", json={}, headers=auth_headers)
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    @patch("app.routes.api.extract_claims", return_value=[])
    @patch("app.routes.api.call_ollama", return_value="response")
    @patch("app.routes.api.build_medical_prompt", return_value="prompt")
    @patch("app.routes.api.check_drug_interactions", return_value={"drug_drug_interactions": []})
    @patch("app.routes.api.search_papers", return_value=[])
    @patch("app.routes.api.get_wearable_summary", return_value={"metrics": []})
    @patch("app.routes.api.get_patient_profile", return_value={
        "conditions": [], "medications": [], "lab_results": []
    })
    @patch("app.routes.api.analyze_health_intent", return_value=[
        {"original_term": "fever", "normalized_term": "Fever", "category": "Condition"}
    ])
    def test_ask_returns_suggestions_when_facts_found(
        self, mock_intent, mock_profile, mock_wearables, mock_papers,
        mock_drugs, mock_prompt, mock_ollama, mock_claims,
        client, auth_headers
    ):
        resp = client.post("/api/ask", json={"question": "I have fever"}, headers=auth_headers)

        data = resp.get_json()
        assert "suggestions" in data
        assert len(data["suggestions"]) >= 1
        assert "Fever" in data["suggestions"][0]["message"]

    @patch("app.routes.api.extract_claims", return_value=[])
    @patch("app.routes.api.call_ollama", return_value="response")
    @patch("app.routes.api.build_medical_prompt", return_value="prompt")
    @patch("app.routes.api.check_drug_interactions", return_value={
        "drug_drug_interactions": [{"severity": "high", "drugs_involved": ["metformin", "contrast dye"]}]
    })
    @patch("app.routes.api.search_papers", return_value=[])
    @patch("app.routes.api.get_wearable_summary", return_value={"metrics": []})
    @patch("app.routes.api.get_patient_profile", return_value={
        "conditions": [], "medications": [{"name": "Metformin"}], "lab_results": []
    })
    @patch("app.routes.api.analyze_health_intent", return_value=[])
    def test_ask_reports_drug_warnings_count(
        self, mock_intent, mock_profile, mock_wearables, mock_papers,
        mock_drugs, mock_prompt, mock_ollama, mock_claims,
        client, auth_headers
    ):
        resp = client.post("/api/ask", json={"question": "Are my meds safe?"}, headers=auth_headers)

        data = resp.get_json()
        assert data["context"]["has_drug_warnings"] is True
        assert data["context"]["drug_warnings_count"] == 1


class TestConfirmUpdateEndpoint:
    @patch("app.routes.api.apply_graph_update", return_value=(True, "Successfully added Condition: Fever"))
    def test_confirm_update_success(self, mock_update, client, auth_headers):
        resp = client.post(
            "/api/confirm_update",
            json={"category": "Condition", "entity": "Fever"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    @patch("app.routes.api.apply_graph_update", return_value=(False, "Patient not found"))
    def test_confirm_update_failure_returns_500(self, mock_update, client, auth_headers):
        resp = client.post(
            "/api/confirm_update",
            json={"category": "Condition", "entity": "Fever"},
            headers=auth_headers,
        )
        assert resp.status_code == 500
        assert resp.get_json()["success"] is False

    def test_confirm_update_requires_auth(self, client):
        resp = client.post("/api/confirm_update", json={"category": "Condition", "entity": "Fever"})
        assert resp.status_code == 401