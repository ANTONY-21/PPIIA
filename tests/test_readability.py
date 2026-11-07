from unittest.mock import patch

import nltk
import ppiia_ai


def test_readability_falls_back_without_network_or_cmudict():
    text = 'The bill creates a board. The board reviews applications from people.'
    with (patch.object(nltk.data, 'find', side_effect=LookupError),
          patch.object(nltk, 'download', side_effect=AssertionError('must not download')),
          patch.object(ppiia_ai.textstat, 'flesch_reading_ease', side_effect=AssertionError('must use fallback'))):
        result = ppiia_ai.compute_metrics(engine='local', model='test', processing_seconds=1,
                                          input_words=30, summary_text=text)
    assert isinstance(result.readability_grade, float)
    assert isinstance(result.flesch_reading_ease, float)
