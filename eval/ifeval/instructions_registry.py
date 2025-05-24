# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""
from eval.ifeval import instructions

# _KEYWORD = "keywords:"

# _LANGUAGE = "language:"

# _LENGTH = "length_constraints:"

# _CONTENT = "detectable_content:"

# _FORMAT = "detectable_format:"

# _MULTITURN = "multi-turn:"

# _COMBINATION = "combination:"

# _STARTEND = "startend:"

# _CHANGE_CASES = "change_case:"

# _PUNCTUATION = "punctuation:"
_LANGUAGE = "language:"
_LINGUISTIC = "linguistic:"
_STRUCTURE = "structure:"
_CONTENT = "content:"
_FREQUENCY = "frequency:"
_FORMAT = "format:"

INSTRUCTION_DICT = {
    _LANGUAGE + "response_language": instructions.ResponseLanguageChecker,

    _LINGUISTIC + "english_capital": instructions.CapitalLettersEnglishChecker,
    _LINGUISTIC + "english_lowercase": instructions.LowercaseLettersEnglishChecker,
    _LINGUISTIC + "no_comma": instructions.CommaChecker,
    _LINGUISTIC + "quotation": instructions.QuotationChecker,
    _LINGUISTIC + "title": instructions.TitleChecker,
    _LINGUISTIC + "number_bullet_lists": instructions.BulletListChecker,
    _LINGUISTIC + "number_highlighted_sections": (instructions.HighlightSectionChecker),

    _STRUCTURE + "number_sentences": instructions.NumberOfSentences,
    _STRUCTURE + "number_words": instructions.NumberOfWords,
    _STRUCTURE + "number_paragraphs": instructions.ParagraphChecker,
    _STRUCTURE + "nth_paragraph_first_word": instructions.ParagraphFirstWordCheck,
    _STRUCTURE + "postscript": instructions.PostscriptChecker,
    _STRUCTURE + "multiple_sections": instructions.SectionChecker,

    _CONTENT + "existence": instructions.KeywordChecker,
    _CONTENT + "forbidden_words": instructions.ForbiddenWords,
    _CONTENT + "named_entity": instructions.NamedEntityChecker,
    _CONTENT + "key_sentences": instructions.KeySentenceChecker,

    _FREQUENCY + "frequency": instructions.KeywordFrequencyChecker,
    _FREQUENCY + "specific_number": instructions.SpecificNumberChecker,
    _FREQUENCY + "letter_frequency": instructions.LetterFrequencyChecker,
    _FREQUENCY + "capital_word_frequency": instructions.CapitalWordFrequencyChecker,

    _FORMAT + "json_format": instructions.JsonFormat,
    _FORMAT + "include_hyperlink": instructions.IncludeHyperlinkChecker,
    _FORMAT + "end_checker": instructions.EndChecker,
    _FORMAT + "question_ending": instructions.QuestionEndingChecker,
    _FORMAT + "number_placeholders": instructions.PlaceholderChecker,


    # _CONTENT + "rephrase_paragraph": instructions.RephraseParagraph,
    # _FORMAT + "constrained_response": instructions.ConstrainedResponseChecker,
    # _FORMAT + "rephrase": instructions.RephraseChecker,
    # _MULTITURN + "constrained_start": instructions.ConstrainedStartChecker,
    #_COMBINATION + "two_responses": instructions.TwoResponsesChecker,
    #_COMBINATION + "repeat_prompt": instructions.RepeatPromptThenAnswer,

}
