# /llm_api/cogniquantum/analyzer.py
"""
プロンプトの複雑性分析モジュール
"""
import logging
import spacy
from typing import Tuple

from .enums import ComplexityRegime

logger = logging.getLogger(__name__)

class AdaptiveComplexityAnalyzer:
    """
    Analyzes the complexity of a prompt to determine the optimal reasoning strategy.
    Enhanced with NLP capabilities if spaCy is available.
    """
    def __init__(self, spacy_model_name="en_core_web_sm"):
        """
        Initializes the analyzer. Tries to load a spaCy model for advanced NLP analysis.
        If spaCy is not installed or the model can't be loaded, it gracefully falls back
        to a keyword-based analysis.
        """
        self.nlp = None
        if spacy:
            try:
                # モデルがシステムに存在するかチェック
                if not spacy.util.is_package(spacy_model_name):
                    logger.info(f"spaCy model '{spacy_model_name}' not found. Attempting to download...")
                    spacy.cli.download(spacy_model_name)
                    logger.info(f"Model '{spacy_model_name}' downloaded successfully.")

                self.nlp = spacy.load(spacy_model_name)
                logger.info(f"spaCy model '{spacy_model_name}' loaded successfully for advanced complexity analysis.")
            except (ImportError, OSError, SystemExit) as e:
                logger.warning(
                    f"Could not load or download spaCy model '{spacy_model_name}'. Error: {e}. "
                    f"Please run 'pip install spacy && python -m spacy download {spacy_model_name}' to enable advanced analysis. "
                    "Falling back to keyword-based analysis."
                )
                self.nlp = None
        else:
            logger.warning(
                "spaCy library not found. "
                "Please run 'pip install spacy' to enable advanced analysis. "
                "Falling back to keyword-based analysis."
            )

    def _keyword_based_analysis(self, prompt: str) -> float:
        """Performs the original keyword-based complexity analysis."""
        logger.debug("Performing keyword-based complexity analysis.")
        prompt_lower = prompt.lower()
        
        # 1. Length Score
        length_score = min(len(prompt.split()) / 5.0, 40)

        # 2. Structure Score
        structural_complexity = 0
        conditional_patterns = ['if', 'when', 'unless', 'provided that', 'given that']
        structural_complexity += sum(prompt_lower.count(p) for p in conditional_patterns) * 3
        hierarchy_indicators = ['first', 'second', 'then', 'next', 'finally', 'step']
        structural_complexity += sum(1 for word in prompt_lower.split() if word in hierarchy_indicators) * 2
        constraint_patterns = ['must', 'cannot', 'should not', 'requires', 'constraint']
        structural_complexity += sum(prompt_lower.count(p) for p in constraint_patterns) * 4
        structure_score = min(structural_complexity, 30)

        # 3. Domain Score
        domain_complexity = 0
        math_keywords = ['calculate', 'solve', 'equation', 'algorithm', 'optimization']
        planning_keywords = ['plan', 'strategy', 'design', 'organize', 'coordinate']
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'assess', 'consider']
        if any(kw in prompt_lower for kw in math_keywords): domain_complexity += 15
        if any(kw in prompt_lower for kw in planning_keywords): domain_complexity += 20
        if any(kw in prompt_lower for kw in analysis_keywords): domain_complexity += 15
        domain_score = min(domain_complexity, 30)

        weights = {'length': 0.2, 'structure': 0.4, 'domain': 0.4}
        total_score = (length_score * weights['length'] +
                       structure_score * weights['structure'] +
                       domain_score * weights['domain'])
        
        return min(max(total_score, 0), 100.0)

    def _nlp_enhanced_analysis(self, prompt: str) -> float:
        """Performs an advanced complexity analysis using spaCy."""
        logger.debug("Performing NLP-enhanced complexity analysis.")
        doc = self.nlp(prompt)

        # 1. Syntactic Complexity
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        if num_sentences == 0: return 5.0
        avg_sent_length = len(doc) / num_sentences
        num_noun_chunks = len(list(doc.noun_chunks))
        syntactic_score = (num_sentences * 1.5) + (avg_sent_length * 0.5) + (num_noun_chunks * 1.0)
        normalized_syntactic = min(syntactic_score / 40.0, 1.0) * 100

        # 2. Lexical Richness
        num_entities = len(doc.ents)
        unique_entity_labels = len(set(ent.label_ for ent in doc.ents))
        entity_score = (num_entities * 2.0) + (unique_entity_labels * 3.0)
        content_words = {token.lemma_.lower() for token in doc if token.pos_ not in ['PUNCT', 'SPACE', 'SYM', 'NUM'] and not token.is_stop}
        lexical_diversity_score = len(content_words) * 0.2
        lexical_score = entity_score + lexical_diversity_score
        normalized_lexical = min(lexical_score / 50.0, 1.0) * 100

        # 3. Cognitive Task Demand
        cognitive_keywords = {'compare', 'contrast', 'analyze', 'evaluate', 'synthesize', 'create', 'argue', 'derive', 'prove'}
        cognitive_lemmas = {token.lemma_.lower() for token in doc if token.pos_ == 'VERB'}
        cognitive_demand_score = len(cognitive_keywords.intersection(cognitive_lemmas)) * 10
        wh_words = {token.lemma_.lower() for token in doc if token.tag_ in ['WDT', 'WP', 'WP$', 'WRB']}
        if wh_words:
            cognitive_demand_score += 15 if 'why' in wh_words or 'how' in wh_words else 5
        normalized_cognitive = min(cognitive_demand_score / 30.0, 1.0) * 100

        weights = {'syntactic': 0.40, 'lexical': 0.35, 'cognitive': 0.25}
        total_score = (normalized_syntactic * weights['syntactic'] +
                       normalized_lexical * weights['lexical'] +
                       normalized_cognitive * weights['cognitive'])

        logger.debug(
            f"NLP Analysis Scores (Normalized): Syntactic={normalized_syntactic:.2f}, "
            f"Lexical={normalized_lexical:.2f}, Cognitive={normalized_cognitive:.2f}"
        )
        return min(max(total_score, 0), 100.0)

    def analyze_complexity(self, prompt: str) -> Tuple[float, ComplexityRegime]:
        """
        Analyzes the prompt's complexity using NLP if available, otherwise falls back
        to a keyword-based method.
        """
        complexity_score: float
        if self.nlp and len(prompt.split()) > 10:
            try:
                logger.info("Performing NLP-enhanced complexity analysis.")
                complexity_score = self._nlp_enhanced_analysis(prompt)
            except Exception as e:
                logger.error(f"An error occurred during NLP analysis: {e}. Falling back to keyword-based method.")
                complexity_score = self._keyword_based_analysis(prompt)
        else:
            if self.nlp:
                logger.info("Prompt is too short for NLP analysis, using keyword-based method.")
            else:
                logger.info("Performing keyword-based complexity analysis.")
            complexity_score = self._keyword_based_analysis(prompt)

        logger.info(f"Calculated complexity score: {complexity_score:.2f}")

        if complexity_score < 30:
            regime = ComplexityRegime.LOW
        elif complexity_score < 70:
            regime = ComplexityRegime.MEDIUM
        else:
            regime = ComplexityRegime.HIGH
        
        logger.info(f"Determined complexity regime: {regime.value}")
        return complexity_score, regime