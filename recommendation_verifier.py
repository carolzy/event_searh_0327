"""
Recommendation Verifier Module

This module provides functionality to verify the accuracy of generated recommendations
by checking event details against their source URLs and detecting potential hallucinations.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import aiohttp
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class RecommendationVerifier:
    """Verifies recommendation data for accuracy and detects hallucinations"""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize the recommendation verifier
        
        Args:
            timeout: Timeout in seconds for HTTP requests
        """
        self.timeout = timeout
        self.stop_words = set(stopwords.words('english'))
        
    async def verify_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Verify a list of recommendations and add verification metadata
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Enhanced recommendations with verification metadata
        """
        verified_recommendations = []
        
        for rec in recommendations:
            # Create a copy of the recommendation to avoid modifying the original
            verified_rec = rec.copy()
            
            # Add verification metadata container
            verified_rec['verification'] = {
                'timestamp': datetime.now().isoformat(),
                'verified_elements': [],
                'hallucination_score': 0.0,
                'confidence_score': 1.0,
                'warnings': []
            }
            
            # Verify company existence
            company_verification = await self.verify_company(verified_rec['name'])
            verified_rec['verification']['verified_elements'].append({
                'element_type': 'company_name',
                'verified': company_verification[0],
                'confidence': company_verification[1],
                'source': company_verification[2]
            })
            
            # Verify events if present
            if 'events' in verified_rec and verified_rec['events']:
                verified_events = await self.verify_events(verified_rec['events'])
                verified_rec['events'] = verified_events
                
                # Add event verification metadata
                for i, event in enumerate(verified_events):
                    if 'verification' in event:
                        verified_rec['verification']['verified_elements'].append({
                            'element_type': f'event_{i}',
                            'verified': event['verification']['verified'],
                            'confidence': event['verification']['confidence'],
                            'source': event['verification'].get('source', '')
                        })
                        
                        # Adjust overall confidence based on event verification
                        if not event['verification']['verified']:
                            verified_rec['verification']['confidence_score'] *= 0.8
                            verified_rec['verification']['hallucination_score'] += 0.2
                            verified_rec['verification']['warnings'].append(
                                f"Event '{event.get('name', 'Unknown')}' could not be verified"
                            )
            
            # Verify news/investment information if present
            if 'recent_news' in verified_rec and verified_rec['recent_news']:
                for i, news in enumerate(verified_rec['recent_news']):
                    if isinstance(news, dict) and 'url' in news and news['url']:
                        news_verification = await self.verify_news_item(news)
                        
                        # Add verification data to the news item
                        news['verification'] = news_verification
                        
                        # Add to overall verification metadata
                        verified_rec['verification']['verified_elements'].append({
                            'element_type': f'news_{i}',
                            'verified': news_verification['verified'],
                            'confidence': news_verification['confidence'],
                            'source': news_verification.get('source', '')
                        })
                        
                        # Adjust overall confidence based on news verification
                        if not news_verification['verified']:
                            verified_rec['verification']['confidence_score'] *= 0.9
                            verified_rec['verification']['hallucination_score'] += 0.1
                            verified_rec['verification']['warnings'].append(
                                f"News item '{news.get('title', 'Unknown')}' could not be verified"
                            )
            
            # Calculate final hallucination score
            hallucination_score = 1.0 - verified_rec['verification']['confidence_score']
            verified_rec['verification']['hallucination_score'] = round(hallucination_score, 2)
            verified_rec['verification']['confidence_score'] = round(verified_rec['verification']['confidence_score'], 2)
            
            # Add hallucination warning if score is high
            if hallucination_score > 0.5:
                verified_rec['verification']['warnings'].append(
                    f"High hallucination score ({hallucination_score:.2f}). This recommendation may contain inaccurate information."
                )
            
            verified_recommendations.append(verified_rec)
        
        return verified_recommendations
    
    async def verify_company(self, company_name: str) -> Tuple[bool, float, str]:
        """
        Verify if a company exists by searching for it
        
        Args:
            company_name: Name of the company to verify
            
        Returns:
            Tuple of (verified, confidence, source)
        """
        # Simple verification using a search engine
        search_url = f"https://www.google.com/search?q={company_name}+company"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Check if company name appears in search results
                        if re.search(company_name, soup.text, re.IGNORECASE):
                            return True, 0.9, search_url
                        else:
                            return False, 0.5, search_url
                    else:
                        logger.warning(f"Failed to verify company {company_name}: HTTP {response.status}")
                        return False, 0.5, ""
        except Exception as e:
            logger.error(f"Error verifying company {company_name}: {str(e)}")
            return False, 0.5, ""
    
    async def verify_events(self, events: List[Dict]) -> List[Dict]:
        """
        Verify event details by checking their URLs
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Enhanced events with verification metadata
        """
        verified_events = []
        
        for event in events:
            # Create a copy of the event to avoid modifying the original
            verified_event = event.copy()
            
            # Skip verification if no URL is provided
            if 'url' not in event or not event['url']:
                verified_event['verification'] = {
                    'verified': False,
                    'confidence': 0.5,
                    'message': "No URL provided for verification"
                }
                verified_events.append(verified_event)
                continue
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(event['url'], timeout=self.timeout) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract event details from the page
                            extracted_data = self._extract_event_data(soup)
                            
                            # Compare extracted data with provided event details
                            verification_result = self._compare_event_data(event, extracted_data)
                            
                            # Add verification metadata
                            verified_event['verification'] = verification_result
                            verified_event['verification']['source'] = event['url']
                        else:
                            logger.warning(f"Failed to verify event {event.get('name')}: HTTP {response.status}")
                            verified_event['verification'] = {
                                'verified': False,
                                'confidence': 0.3,
                                'message': f"Failed to access URL: HTTP {response.status}"
                            }
            except Exception as e:
                logger.error(f"Error verifying event {event.get('name')}: {str(e)}")
                verified_event['verification'] = {
                    'verified': False,
                    'confidence': 0.3,
                    'message': f"Error accessing URL: {str(e)}"
                }
            
            verified_events.append(verified_event)
        
        return verified_events
    
    def _extract_event_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract event data from HTML content
        
        Args:
            soup: BeautifulSoup object of the event page
            
        Returns:
            Dictionary of extracted event data
        """
        extracted_data = {
            'name': None,
            'date': None,
            'location': None,
            'description': None,
            'attendees': []
        }
        
        # Extract event name (usually in h1 or h2 tags)
        title_tags = soup.find_all(['h1', 'h2', 'h3'])
        if title_tags:
            extracted_data['name'] = title_tags[0].text.strip()
        
        # Extract date (look for date patterns)
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD/MM/YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Month YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b'  # Month DD, YYYY
        ]
        
        text = soup.text
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                extracted_data['date'] = date_match.group(0)
                break
        
        # Extract location
        location_patterns = [
            r'\b(?:in|at|location)\s*:\s*([^,\.]+)',
            r'\b([^,\.]+)\s+Convention Center\b',
            r'\b([^,\.]+)\s+Conference Center\b',
            r'\b([^,\.]+)\s+Hotel\b'
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, text, re.IGNORECASE)
            if location_match:
                extracted_data['location'] = location_match.group(1).strip()
                break
        
        # Extract description (paragraphs near the title)
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Get the first substantial paragraph (more than 50 chars)
            for p in paragraphs:
                if len(p.text.strip()) > 50:
                    extracted_data['description'] = p.text.strip()
                    break
        
        # Extract potential attendees/speakers
        speaker_sections = soup.find_all(['div', 'section'], class_=lambda c: c and any(x in c.lower() for x in ['speaker', 'presenter', 'attendee']))
        if speaker_sections:
            for section in speaker_sections:
                names = section.find_all(['h3', 'h4', 'strong'])
                for name in names:
                    extracted_data['attendees'].append(name.text.strip())
        
        return extracted_data
    
    def _compare_event_data(self, event: Dict, extracted_data: Dict) -> Dict:
        """
        Compare provided event data with extracted data
        
        Args:
            event: Event dictionary from recommendation
            extracted_data: Dictionary of extracted event data
            
        Returns:
            Verification result dictionary
        """
        verification_result = {
            'verified': False,
            'confidence': 0.0,
            'matches': {},
            'mismatches': {}
        }
        
        # Initialize confidence score
        confidence = 0.0
        match_count = 0
        
        # Compare event name
        if extracted_data['name'] and event.get('name'):
            similarity = self._calculate_text_similarity(extracted_data['name'], event['name'])
            if similarity > 0.6:
                verification_result['matches']['name'] = similarity
                confidence += similarity
                match_count += 1
            else:
                verification_result['mismatches']['name'] = similarity
        
        # Compare date
        if extracted_data['date'] and event.get('date'):
            # Normalize dates for comparison
            similarity = self._calculate_text_similarity(extracted_data['date'], event['date'])
            if similarity > 0.6:
                verification_result['matches']['date'] = similarity
                confidence += similarity
                match_count += 1
            else:
                verification_result['mismatches']['date'] = similarity
        
        # Compare location
        if extracted_data['location'] and event.get('location'):
            similarity = self._calculate_text_similarity(extracted_data['location'], event['location'])
            if similarity > 0.6:
                verification_result['matches']['location'] = similarity
                confidence += similarity
                match_count += 1
            else:
                verification_result['mismatches']['location'] = similarity
        
        # Compare description
        if extracted_data['description'] and event.get('description'):
            similarity = self._calculate_text_similarity(extracted_data['description'], event['description'])
            if similarity > 0.4:  # Lower threshold for description
                verification_result['matches']['description'] = similarity
                confidence += similarity
                match_count += 1
            else:
                verification_result['mismatches']['description'] = similarity
        
        # Calculate final confidence score
        if match_count > 0:
            final_confidence = confidence / match_count
            
            # Event is verified if confidence is above threshold and at least 2 fields match
            verification_result['verified'] = final_confidence > 0.6 and match_count >= 2
            verification_result['confidence'] = round(final_confidence, 2)
            
            # Add verification message
            if verification_result['verified']:
                verification_result['message'] = f"Event verified with {final_confidence:.2f} confidence"
            else:
                verification_result['message'] = f"Event verification failed with {final_confidence:.2f} confidence"
        else:
            verification_result['message'] = "No matching fields found for verification"
            verification_result['confidence'] = 0.3
        
        return verification_result
    
    async def verify_news_item(self, news: Dict) -> Dict:
        """
        Verify a news item by checking its URL
        
        Args:
            news: News dictionary from recommendation
            
        Returns:
            Verification result dictionary
        """
        verification_result = {
            'verified': False,
            'confidence': 0.0,
            'source': news.get('url', '')
        }
        
        # Skip verification if no URL is provided
        if 'url' not in news or not news['url']:
            verification_result['message'] = "No URL provided for verification"
            verification_result['confidence'] = 0.5
            return verification_result
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(news['url'], timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract article text
                        article_text = self._extract_article_text(soup)
                        
                        # Check if the summary content is present in the article
                        if news.get('summary'):
                            # For each sentence in the summary, check if it appears in the article
                            sentences = sent_tokenize(news['summary'])
                            matched_sentences = 0
                            
                            for sentence in sentences:
                                # Skip very short sentences
                                if len(sentence.split()) < 5:
                                    continue
                                    
                                similarity = self._calculate_text_similarity(article_text, sentence)
                                if similarity > 0.7:
                                    matched_sentences += 1
                            
                            # Calculate verification confidence
                            if len(sentences) > 0:
                                verification_confidence = matched_sentences / len(sentences)
                                verification_result['verified'] = verification_confidence > 0.5
                                verification_result['confidence'] = round(verification_confidence, 2)
                                
                                if verification_result['verified']:
                                    verification_result['message'] = f"News content verified with {verification_confidence:.2f} confidence"
                                else:
                                    verification_result['message'] = f"News content verification failed with {verification_confidence:.2f} confidence"
                            else:
                                verification_result['message'] = "No sentences to verify in summary"
                                verification_result['confidence'] = 0.5
                        else:
                            verification_result['message'] = "No summary provided for verification"
                            verification_result['confidence'] = 0.5
                    else:
                        verification_result['message'] = f"Failed to access URL: HTTP {response.status}"
                        verification_result['confidence'] = 0.3
        except Exception as e:
            verification_result['message'] = f"Error accessing URL: {str(e)}"
            verification_result['confidence'] = 0.3
        
        return verification_result
    
    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """
        Extract article text from HTML content
        
        Args:
            soup: BeautifulSoup object of the article page
            
        Returns:
            Extracted article text
        """
        # Try to find article content in common article containers
        article_container = soup.find(['article', 'main', 'div'], class_=lambda c: c and any(x in str(c).lower() for x in ['article', 'content', 'post']))
        
        if article_container:
            # Get all paragraphs in the article container
            paragraphs = article_container.find_all('p')
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all('p')
        
        # Combine paragraphs into a single text
        article_text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 20])
        
        return article_text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Tokenize
        words1 = set(word for word in re.findall(r'\b\w+\b', text1) if word not in self.stop_words)
        words2 = set(word for word in re.findall(r'\b\w+\b', text2) if word not in self.stop_words)
        
        # Handle empty sets
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


async def verify_recommendations(recommendations: List[Dict]) -> List[Dict]:
    """
    Convenience function to verify recommendations
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Enhanced recommendations with verification metadata
    """
    verifier = RecommendationVerifier()
    return await verifier.verify_recommendations(recommendations)
