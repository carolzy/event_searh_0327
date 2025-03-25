import os
import logging
import json
import random
from datetime import datetime
import httpx
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CompanyRecommender:
    """Recommends target companies based on user preferences"""
    
    def __init__(self, flow_controller):
        """Initialize the company recommender"""
        self.flow_controller = flow_controller
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_llm = self.gemini_api_key is not None or self.perplexity_api_key is not None or self.openai_api_key is not None
        self.use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
        
        # Priority sources for company information
        self.priority_news_sources = [
            "techcrunch.com",
            "crunchbase.com",
            "pitchbook.com",
            "venturebeat.com",
            "forbes.com",
            "businessinsider.com",
            "cnbc.com",
            "reuters.com",
            "bloomberg.com",
            "wsj.com"
        ]
        
        # Priority sources for events
        self.priority_event_sources = [
            "eventbrite.com",
            "luma.events",
            "meetup.com",
            "conference.com",
            "summit.com"
        ]
        
        # Get or create user memory
        self.user_id = flow_controller.user_id if hasattr(flow_controller, 'user_id') else "default_user"
        
        # Initialize user memory if the module is available
        try:
            from user_memory import UserMemory
            self.user_memory = UserMemory(self.user_id)
            self.user_memory_available = True
        except ImportError:
            logger.warning("UserMemory module not available. User preferences will not be applied.")
            self.user_memory_available = False
        
        if not self.use_llm and not self.use_mock_data:
            logger.warning("No API keys found for LLM and mock data is disabled. Using mock data anyway as fallback.")
    
    async def generate_recommendations(self, count=3, verify=True):
        """Generate company recommendations based on user preferences."""
        try:
            logger.info("Generating company recommendations...")
            
            # If mock data is enabled, use that directly
            if self.use_mock_data:
                logger.info("Using mock data for recommendations as configured")
                recommendations = self._get_mock_recommendations(count)
                return recommendations
            
            # Get user preferences from flow controller with better error handling
            product = self._get_flow_controller_attribute('product', '')
            market = self._get_flow_controller_attribute('market', '')
            company_size = self._get_flow_controller_attribute('company_size', '')
            zip_code = self._get_flow_controller_attribute('location', '')
            linkedin_consent = self._get_flow_controller_attribute('linkedin_consent', False)
            keywords = self._get_flow_controller_attribute('keywords', [])
            
            # Log input data
            logger.info(f"Recommendation inputs: product='{product}', market='{market}', company_size='{company_size}', zip_code='{zip_code}', linkedin_consent={linkedin_consent}, keywords={keywords}")
            
            # Only try LLM-based recommendations if API keys are available
            if not self.use_llm:
                logger.warning("No LLM API keys available. Using mock recommendations.")
                return self._get_mock_recommendations(count)
            
            # Generate recommendations using Gemini
            try:
                recommendations = await self._generate_with_gemini(
                    product=product,
                    market=market,
                    company_size=company_size,
                    zip_code=zip_code,
                    keywords=keywords,
                    linkedin_consent=linkedin_consent,
                    count=count
                )
                
                # If we got empty recommendations, fall back to mock data
                if not recommendations:
                    logger.warning("Received empty recommendations from LLM. Using mock data.")
                    return self._get_mock_recommendations(count)
                
            except Exception as e:
                logger.error(f"Error generating recommendations with LLM: {str(e)}")
                logger.info("Falling back to mock recommendations")
                return self._get_mock_recommendations(count)
            
            # Verify recommendations if requested
            if verify and recommendations:
                logger.info(f"Verifying {len(recommendations)} recommendations...")
                verified_recommendations = []
                for rec in recommendations:
                    if self._verify_recommendation(rec):
                        verified_recommendations.append(rec)
                    else:
                        logger.warning(f"Removed invalid recommendation: {rec.get('name', 'Unknown')}")
                
                # If verification removed all recommendations, fall back to mock data
                if not verified_recommendations:
                    logger.warning("All recommendations were removed during verification. Using mock data.")
                    return self._get_mock_recommendations(count)
                
                recommendations = verified_recommendations
                logger.info(f"Verified {len(recommendations)} recommendations")
            
            # Apply user preferences if UserMemory is available
            if self.user_memory_available:
                try:
                    recommendations = self.user_memory.apply_preferences_to_recommendations(recommendations)
                    logger.info("Applied user preferences to recommendations")
                except Exception as e:
                    logger.error(f"Error applying user preferences: {str(e)}")
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Unexpected error generating recommendations: {str(e)}")
            logger.info("Falling back to mock recommendations due to unexpected error")
            return self._get_mock_recommendations(count)
    
    def _get_flow_controller_attribute(self, attribute, default=None):
        """Safely get an attribute from the flow controller"""
        # First try the get_X method
        getter_method = f'get_{attribute}'
        if hasattr(self.flow_controller, getter_method):
            try:
                getter = getattr(self.flow_controller, getter_method)
                return getter()
            except Exception as e:
                logger.error(f"Error calling {getter_method}(): {str(e)}")
        
        # Then try direct attribute access
        if hasattr(self.flow_controller, attribute):
            try:
                return getattr(self.flow_controller, attribute)
            except Exception as e:
                logger.error(f"Error accessing attribute {attribute}: {str(e)}")
                
        # Then try current_X attribute (used in some places)
        current_attribute = f'current_{attribute}'
        if hasattr(self.flow_controller, current_attribute):
            try:
                return getattr(self.flow_controller, current_attribute)
            except Exception as e:
                logger.error(f"Error accessing attribute {current_attribute}: {str(e)}")
        
        # Fall back to default
        logger.warning(f"Could not get {attribute} from flow controller. Using default: {default}")
        return default
    
    def _verify_recommendation(self, recommendation):
        """
        Verify that a recommendation is valid and contains all required fields.
        
        Args:
            recommendation: The recommendation to verify
            
        Returns:
            bool: True if the recommendation is valid, False otherwise
        """
        # Check if recommendation is a dictionary
        if not isinstance(recommendation, dict):
            logger.error("Recommendation is not a dictionary")
            return False
        
        # Check for required fields
        required_fields = ['name']
        for field in required_fields:
            if field not in recommendation:
                logger.error(f"Recommendation missing required field: {field}")
                return False
            if not recommendation[field]:
                logger.error(f"Recommendation has empty required field: {field}")
                return False
        
        # Add default fields if missing
        if 'description' not in recommendation or not recommendation['description']:
            recommendation['description'] = f"A company in the {self._get_flow_controller_attribute('market', 'technology')} sector."
            
        if 'reason' not in recommendation:
            recommendation['reason'] = f"Matches your {self._get_flow_controller_attribute('product', 'product')} offering."
            
        if 'match_score' not in recommendation:
            recommendation['match_score'] = random.randint(70, 95)
        
        return True
    
    async def _generate_with_gemini(self, product, market, company_size, zip_code, keywords, linkedin_consent, count):
        """Generate recommendations using the Gemini API"""
        try:
            # Construct a prompt based on user preferences
            prompt = self._construct_recommendation_prompt(product, market, company_size, zip_code, keywords, linkedin_consent)
            
            # Call the Gemini API with optimized settings
            async with httpx.AsyncClient(timeout=90.0) as client:  
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
                
                # Check if we need to use a more capable model for complex queries
                use_pro_model = False
                tech_terms = ["gemini", "flash", "2.0", "ai", "ml", "llm", "gpt", "claude", "anthropic", "openai"]
                startup_terms = ["startup", "early stage", "seed", "series a", "emerging"]
                
                # Use Pro model for more complex queries about startups or specific technologies
                if (product and any(term in product.lower() for term in tech_terms + startup_terms)) or (keywords and any(term in " ".join(keywords).lower() for term in tech_terms + startup_terms)):
                    use_pro_model = True
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro:generateContent?key={self.gemini_api_key}"
                    logger.info("Using Gemini 2.0 Pro model for more detailed startup/technology search")
                
                data = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.2 if not use_pro_model else 0.4,  # Higher temperature for more diverse results with Pro
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 4096 if not use_pro_model else 8192  # Increased token limit for Pro model
                    }
                }
                
                logger.info(f"Calling Gemini {'2.0 Pro' if use_pro_model else '2.0 Flash'} API for recommendations")
                
                try:
                    response = await client.post(
                        url,
                        json=data,
                        timeout=90.0  # Increased timeout for more detailed responses
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info("Received response from Gemini API")
                        
                        if "candidates" in result and len(result["candidates"]) > 0:
                            content = result["candidates"][0]["content"]
                            if "parts" in content and len(content["parts"]) > 0:
                                recommendations_text = content["parts"][0]["text"]
                                logger.info(f"Raw recommendations text length: {len(recommendations_text)}")
                                
                                # Parse the recommendations from the response
                                try:
                                    recommendations = self._parse_recommendations_from_llm_response(recommendations_text)
                                    logger.info(f"Successfully parsed {len(recommendations)} recommendations")
                                    
                                    # Return the recommendations
                                    return recommendations[:count]
                                except Exception as e:
                                    logger.error(f"Error parsing recommendations: {str(e)}")
                                    logger.error(f"Raw response: {recommendations_text[:500]}...")
                                    # Instead of raising, return empty list to trigger fallback
                                    return []
                        else:
                            logger.error(f"Unexpected response format from Gemini API: {result}")
                            return []
                    else:
                        logger.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                        return []
                        
                except httpx.RequestError as e:
                    logger.error(f"Request error calling Gemini API: {str(e)}")
                    return []
                except httpx.TimeoutException as e:
                    logger.error(f"Timeout calling Gemini API: {str(e)}")
                    return []
                        
        except Exception as e:
            logger.error(f"Error generating recommendations with Gemini: {str(e)}")
            return []
    
    def _construct_recommendation_prompt(self, product, market, company_size, zip_code, keywords, linkedin_consent):
        """Construct a prompt for the LLM to generate company recommendations"""
        # Format keywords as a comma-separated list
        keywords_context = ", ".join(keywords) if keywords else "No specific keywords provided"
        
        # Add location context if available
        location_context = f"LOCATION: {zip_code}" if zip_code else "LOCATION: Not specified"
        
        # Add LinkedIn context if available
        linkedin_context = "LinkedIn data is available for network-based recommendations." if linkedin_consent else "LinkedIn data is not available."
        
        # Get user preference context from memory
        user_preference_context = ""
        if self.user_memory_available:
            try:
                user_preference_context = self.user_memory.get_llm_preference_prompt() 
            except Exception as e:
                logger.error(f"Error getting user preferences for prompt: {str(e)}")
        
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if we're looking for startups specifically
        startup_focus = ""
        if product and "startup" in product.lower():
            startup_focus = "\nIMPORTANT: Focus specifically on EARLY-STAGE STARTUPS and EMERGING COMPANIES rather than established enterprises."
        elif company_size and any(term in company_size.lower() for term in ["small", "startup", "early", "seed", "series a"]):
            startup_focus = "\nIMPORTANT: Focus specifically on EARLY-STAGE STARTUPS and EMERGING COMPANIES rather than established enterprises."
        elif keywords and any(term in " ".join(keywords).lower() for term in ["startup", "early stage", "seed", "series a", "emerging"]):
            startup_focus = "\nIMPORTANT: Focus specifically on EARLY-STAGE STARTUPS and EMERGING COMPANIES rather than established enterprises."
        
        # Check if we're looking for companies using specific technologies
        tech_focus = ""
        tech_terms = ["gemini", "flash", "2.0", "ai", "ml", "llm", "gpt", "claude", "anthropic", "openai"]
        if product and any(term in product.lower() for term in tech_terms):
            tech_focus = f"\nIMPORTANT: Focus on companies that are actively using or developing {product} technology."
        elif keywords and any(term in " ".join(keywords).lower() for term in tech_terms):
            matching_terms = [term for term in tech_terms if term in " ".join(keywords).lower()]
            tech_focus = f"\nIMPORTANT: Focus on companies that are actively using or developing {', '.join(matching_terms)} technology."
        
        return f"""You are a financial analyst specializing in B2B company research. Generate TARGET company recommendations for a B2B sales professional with the following profile:

PRODUCT/SERVICE: {product}
TARGET MARKET/INDUSTRY: {market}
TARGET COMPANY SIZE: {company_size}
KEYWORDS: {keywords_context}
{location_context}
{linkedin_context}

CURRENT DATE: {current_date}

{user_preference_context}{startup_focus}{tech_focus}

IMPORTANT CLARIFICATION: The user is selling {product} to companies in the {market} market. I need you to recommend POTENTIAL CUSTOMER COMPANIES that the user could sell to, NOT competitors who offer similar products. Focus on companies that might NEED or BUY {product}.

IMPORTANT: Focus on REAL companies only. DO NOT make up or hallucinate information. If you're uncertain about details, provide less information rather than inventing facts. Only include information you are confident is accurate.

For each company, you MUST provide ALL of the following information:
1. Company name (must be a real company)
2. Website URL (must be a real website)
3. Industry (specific industry the company operates in)
4. Company size (employees or revenue)
5. Brief description (1-2 sentences about what they actually do)
6. Current year's investment areas and focus (list at least 3 specific areas)
7. Budget allocation information (how they're allocating resources)
8. 2-3 recent news articles with direct quotes from executives (include the source, date, and URL for each)
9. 3-5 key leads/decision makers with titles, emails, and LinkedIn profiles
10. Upcoming events where company representatives will be present (include date, location, and URL)

Take your time to provide detailed, high-quality recommendations. Quality is more important than speed.

Format each recommendation as a JSON object with the following structure:
{
  "name": "Company Name",
  "website": "https://company-website.com",
  "industry": "Industry",
  "size": "Size (employees/revenue)",
  "description": "Brief description",
  "investment_areas": ["Area 1", "Area 2", "Area 3"],
  "budget_allocation": "Budget allocation details",
  "articles": [
    {
      "title": "Article Title",
      "source": "Source Name",
      "date": "Publication Date",
      "url": "https://article-url.com",
      "quote": "Direct quote from executive"
    }
  ],
  "leads": [
    {
      "name": "Lead Name",
      "title": "Job Title",
      "email": "email@company.com",
      "linkedin": "https://linkedin.com/in/profile"
    }
  ],
  "events": [
    {
      "name": "Event Name",
      "date": "Event Date",
      "location": "Event Location",
      "url": "https://event-url.com",
      "description": "Brief description of the event and why it's relevant",
      "attending_companies": ["Company 1", "Company 2"]
    }
  ]
}

Return your response as a valid JSON array of company objects. Include at least 3 detailed company recommendations.
"""
    
    def _parse_recommendations_from_llm_response(self, response: str) -> List[Dict]:
        """Parse recommendations from LLM response"""
        try:
            # Log hash of response for debugging
            response_hash = hash(response)
            logger.info(f"Parsing recommendations from response hash: {response_hash}")
            
            # First, check if the entire response is valid JSON
            try:
                recommendations = json.loads(response)
                if isinstance(recommendations, list):
                    logger.info(f"Successfully parsed response as JSON array with {len(recommendations)} recommendations")
                    return recommendations
                elif isinstance(recommendations, dict):
                    logger.info("Successfully parsed response as single JSON object")
                    return [recommendations]
            except json.JSONDecodeError:
                # Not valid JSON, continue with extraction methods
                pass
            
            # Clean up the response to extract just the JSON part
            # First, try to find JSON array in the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Extract the JSON array
                json_str = response[json_start:json_end]
                try:
                    recommendations = json.loads(json_str)
                    if isinstance(recommendations, list):
                        logger.info(f"Successfully extracted JSON array with {len(recommendations)} recommendations")
                        return recommendations
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extracted JSON array. Attempting other methods.")
            
            # If we couldn't find a JSON array, look for JSON objects
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Extract the JSON object and wrap it in an array
                json_str = response[json_start:json_end]
                try:
                    recommendation = json.loads(json_str)
                    if isinstance(recommendation, dict):
                        logger.info("Successfully extracted a single JSON object recommendation")
                        return [recommendation]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extracted JSON object. Attempting other methods.")
            
            # If we still couldn't find valid JSON, try to extract code blocks
            code_block_patterns = [
                r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON code blocks
                r'```\s*([\s\S]*?)\s*```',      # Generic code blocks
                r'<json>\s*([\s\S]*?)\s*</json>'  # XML-style JSON blocks
            ]
            
            for pattern in code_block_patterns:
                import re
                json_blocks = re.findall(pattern, response)
                if json_blocks:
                    for block in json_blocks:
                        try:
                            content = json.loads(block)
                            if isinstance(content, list):
                                logger.info(f"Successfully extracted JSON array from code block with {len(content)} recommendations")
                                return content
                            elif isinstance(content, dict):
                                logger.info("Successfully extracted a single JSON object from code block")
                                return [content]
                        except json.JSONDecodeError:
                            continue
            
            # Try to fix common JSON errors and retry
            fixed_response = self._fix_common_json_errors(response)
            if fixed_response != response:
                try:
                    recommendations = json.loads(fixed_response)
                    if isinstance(recommendations, list):
                        logger.info(f"Successfully parsed fixed JSON with {len(recommendations)} recommendations")
                        return recommendations
                    elif isinstance(recommendations, dict):
                        logger.info("Successfully parsed fixed JSON as single object")
                        return [recommendations]
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, extract structured data using regex patterns
            companies = self._extract_companies_with_regex(response)
            if companies:
                logger.info(f"Extracted {len(companies)} companies using regex patterns")
                return companies
            
            # Last resort: generate mock data with relevant company names if possible
            logger.error(f"Could not parse recommendations from response: {response[:500]}...")
            companies = self._extract_company_names(response)
            if companies:
                logger.info(f"Generating mock data for {len(companies)} extracted company names")
                return self._generate_mock_data_for_companies(companies)
            
            # If absolutely everything fails, return an empty list to trigger fallback
            logger.error("All parsing methods failed. Returning empty list.")
            return []
            
        except Exception as e:
            logger.error(f"Error parsing recommendations: {str(e)}")
            logger.error(f"Response: {response[:500]}...")
            return []
    
    def _fix_common_json_errors(self, json_str):
        """Fix common JSON errors in the response"""
        # Replace single quotes with double quotes
        fixed = json_str.replace("'", '"')
        
        # Fix trailing commas in arrays and objects
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed = re.sub(r',\s*}', '}', fixed)
        
        # Fix missing quotes around keys
        fixed = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', fixed)
        
        return fixed
    
    def _extract_companies_with_regex(self, text):
        """Extract company data using regex patterns as a last resort"""
        companies = []
        
        # Try to find company blocks
        company_blocks = re.findall(r'(?:Company|Name):\s*([^"\n]+)', text)
        
        for i, company_name in enumerate(company_blocks):
            company_name = company_name.strip().strip(',:')
            if not company_name:
                continue
                
            # Extract a description if possible
            description_match = re.search(rf'{re.escape(company_name)}.*?(?:Description|About):\s*([^"\n]+)', text, re.DOTALL)
            description = description_match.group(1).strip() if description_match else f"A company that may be interested in {self._get_flow_controller_attribute('product', 'this product')}."
            
            # Create a basic company object
            company = {
                "name": company_name,
                "description": description,
                "match_score": random.randint(70, 95),
                "reason": f"Matches your target market in {self._get_flow_controller_attribute('market', 'technology')}."
            }
            
            companies.append(company)
        
        return companies
    
    def _extract_company_names(self, text):
        """Extract just company names from the text"""
        # Look for patterns like "Company Name: X" or "1. X" or "- X"
        company_patterns = [
            r'(?:Company|Name):\s*([^"\n,]+)',
            r'\d+\.\s+([^"\n,]+)',
            r'-\s+([^"\n,]+)'
        ]
        
        companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend([m.strip() for m in matches if m.strip()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_companies = [c for c in companies if not (c in seen or seen.add(c))]
        
        return unique_companies
    
    def _generate_mock_data_for_companies(self, company_names):
        """Generate mock data for extracted company names"""
        mock_recommendations = []
        
        for name in company_names[:10]:  # Limit to 10 companies
            mock_recommendations.append({
                "name": name,
                "description": f"A company in the {self._get_flow_controller_attribute('market', 'technology')} sector.",
                "match_score": random.randint(70, 95),
                "reason": f"Matches your {self._get_flow_controller_attribute('product', 'product')} offering."
            })
        
        return mock_recommendations
    
    def _get_mock_recommendations(self, count=5):
        """Generate mock recommendations for testing or when API calls fail"""
        logger.info(f"Generating {count} mock recommendations")
        
        # Sample company data
        companies = [
            {
                "name": "TechNova Solutions",
                "website": "https://technova.ai",
                "industry": "Enterprise Software",
                "size": "500-1000 employees",
                "description": "Leading provider of AI-powered business intelligence solutions for mid-market companies.",
                "investment_areas": ["AI/ML Infrastructure", "Data Analytics", "Cloud Migration"],
                "budget_allocation": "40% R&D, 30% Sales & Marketing, 20% Operations, 10% Other",
                "match_score": 95,
                "reason": "Strong match for AI data platforms targeting mid-market customers",
                "articles": [
                    {
                        "title": "TechNova Secures $50M Series C Funding",
                        "source": "TechCrunch",
                        "date": "March 15, 2025",
                        "url": "https://techcrunch.com/2025/03/15/technova-funding",
                        "quote": "Our focus this year is on expanding our AI capabilities and helping more mid-market companies leverage data for growth."
                    },
                    {
                        "title": "The Future of Business Intelligence",
                        "source": "Forbes",
                        "date": "February 28, 2025",
                        "url": "https://forbes.com/future-bi-2025",
                        "quote": "We're seeing a massive shift toward predictive analytics among our customer base. Companies want to know not just what happened, but what will happen next."
                    }
                ],
                "leads": [
                    {
                        "name": "Sarah Johnson",
                        "title": "Chief Technology Officer",
                        "email": "sjohnson@technova.ai",
                        "linkedin": "https://linkedin.com/in/sarahjohnson"
                    },
                    {
                        "name": "Michael Chen",
                        "title": "VP of Product",
                        "email": "mchen@technova.ai",
                        "linkedin": "https://linkedin.com/in/michaelchen"
                    },
                    {
                        "name": "Jessica Williams",
                        "title": "Director of Data Science",
                        "email": "jwilliams@technova.ai",
                        "linkedin": "https://linkedin.com/in/jessicawilliams"
                    }
                ],
                "events": [
                    {
                        "name": "Enterprise AI Summit 2025",
                        "date": "April 10-12, 2025",
                        "location": "San Francisco, CA",
                        "url": "https://enterpriseaisummit.com",
                        "description": "Annual conference focusing on enterprise AI adoption",
                        "attending_companies": ["TechNova", "Microsoft", "Google", "Amazon"]
                    },
                    {
                        "name": "Data Analytics World",
                        "date": "May 15-17, 2025",
                        "location": "Chicago, IL",
                        "url": "https://dataanalyticsworld.com",
                        "description": "Conference on data analytics and business intelligence",
                        "attending_companies": ["TechNova", "Tableau", "Snowflake", "Databricks"]
                    }
                ]
            },
            {
                "name": "CloudSecure Inc.",
                "website": "https://cloudsecure.io",
                "industry": "Cybersecurity",
                "size": "200-500 employees",
                "description": "Innovative cloud security platform protecting enterprise data across multi-cloud environments.",
                "investment_areas": ["Zero Trust Architecture", "Cloud Security", "Threat Intelligence"],
                "budget_allocation": "35% R&D, 25% Sales & Marketing, 30% Operations, 10% Other",
                "match_score": 90,
                "reason": "Ideal for security solutions targeting cloud infrastructure",
                "articles": [
                    {
                        "title": "CloudSecure Launches New Zero Trust Platform",
                        "source": "VentureBeat",
                        "date": "January 20, 2025",
                        "url": "https://venturebeat.com/cloudsecure-zero-trust",
                        "quote": "Zero Trust isn't just a buzzword anymore—it's a necessity for any organization serious about security in today's distributed work environment."
                    },
                    {
                        "title": "The State of Cloud Security in 2025",
                        "source": "CyberWire",
                        "date": "March 5, 2025",
                        "url": "https://cyberwire.com/cloud-security-2025",
                        "quote": "We're seeing a 300% increase in sophisticated attacks targeting cloud infrastructure. Companies need to rethink their security posture from the ground up."
                    }
                ],
                "leads": [
                    {
                        "name": "David Rodriguez",
                        "title": "Chief Security Officer",
                        "email": "drodriguez@cloudsecure.io",
                        "linkedin": "https://linkedin.com/in/davidrodriguez"
                    },
                    {
                        "name": "Aisha Patel",
                        "title": "VP of Engineering",
                        "email": "apatel@cloudsecure.io",
                        "linkedin": "https://linkedin.com/in/aishapatel"
                    },
                    {
                        "name": "Thomas Wright",
                        "title": "Director of Cloud Operations",
                        "email": "twright@cloudsecure.io",
                        "linkedin": "https://linkedin.com/in/thomaswright"
                    }
                ],
                "events": [
                    {
                        "name": "RSA Conference 2025",
                        "date": "April 25-29, 2025",
                        "location": "San Francisco, CA",
                        "url": "https://rsaconference.com",
                        "description": "World's leading cybersecurity conference",
                        "attending_companies": ["CloudSecure", "CrowdStrike", "Palo Alto Networks", "Fortinet"]
                    }
                ]
            },
            {
                "name": "FinEdge Systems",
                "website": "https://finedge.com",
                "industry": "Financial Technology",
                "size": "100-200 employees",
                "description": "Next-generation payment processing and financial analytics platform for SMBs.",
                "investment_areas": ["Payment Processing", "Fraud Detection", "Financial Analytics"],
                "budget_allocation": "30% R&D, 40% Sales & Marketing, 20% Operations, 10% Other",
                "match_score": 85,
                "reason": "Perfect match for fintech targeting payment solutions",
                "articles": [
                    {
                        "title": "FinEdge Expands SMB Payment Solutions",
                        "source": "CNBC",
                        "date": "February 10, 2025",
                        "url": "https://cnbc.com/finedge-expansion",
                        "quote": "Small businesses have been underserved by traditional payment processors for too long. We're changing that with transparent pricing and modern APIs."
                    }
                ],
                "leads": [
                    {
                        "name": "Jennifer Kim",
                        "title": "CEO",
                        "email": "jkim@finedge.com",
                        "linkedin": "https://linkedin.com/in/jenniferkim"
                    },
                    {
                        "name": "Robert Garcia",
                        "title": "Head of Sales",
                        "email": "rgarcia@finedge.com",
                        "linkedin": "https://linkedin.com/in/robertgarcia"
                    }
                ],
                "events": [
                    {
                        "name": "Money 20/20",
                        "date": "June 5-8, 2025",
                        "location": "Las Vegas, NV",
                        "url": "https://money2020.com",
                        "description": "World's largest fintech event",
                        "attending_companies": ["FinEdge", "Stripe", "Square", "PayPal"]
                    }
                ]
            },
            {
                "name": "GreenScale Technologies",
                "website": "https://greenscale.tech",
                "industry": "CleanTech",
                "size": "50-100 employees",
                "description": "Sustainable energy management solutions for commercial buildings and industrial facilities.",
                "investment_areas": ["Energy Efficiency", "Smart Buildings", "Carbon Footprint Reduction"],
                "budget_allocation": "45% R&D, 20% Sales & Marketing, 25% Operations, 10% Other",
                "match_score": 82,
                "reason": "Strong fit for energy tech solutions",
                "articles": [
                    {
                        "title": "GreenScale Reduces Carbon Footprint for Fortune 500 Clients",
                        "source": "Bloomberg",
                        "date": "March 22, 2025",
                        "url": "https://bloomberg.com/greenscale-carbon",
                        "quote": "Our clients are seeing an average of 32% reduction in energy costs while meeting their sustainability goals ahead of schedule."
                    }
                ],
                "leads": [
                    {
                        "name": "Emma Wilson",
                        "title": "Chief Sustainability Officer",
                        "email": "ewilson@greenscale.tech",
                        "linkedin": "https://linkedin.com/in/emmawilson"
                    },
                    {
                        "name": "James Thompson",
                        "title": "VP of Business Development",
                        "email": "jthompson@greenscale.tech",
                        "linkedin": "https://linkedin.com/in/jamesthompson"
                    }
                ],
                "events": [
                    {
                        "name": "Sustainable Business Summit",
                        "date": "May 20-22, 2025",
                        "location": "Boston, MA",
                        "url": "https://sustainablebusinesssummit.com",
                        "description": "Conference focused on sustainable business practices",
                        "attending_companies": ["GreenScale", "Tesla", "Siemens", "Schneider Electric"]
                    }
                ]
            },
            {
                "name": "HealthSync",
                "website": "https://healthsync.io",
                "industry": "Healthcare Technology",
                "size": "200-500 employees",
                "description": "AI-powered healthcare coordination platform improving patient outcomes and reducing administrative costs.",
                "investment_areas": ["Patient Engagement", "Clinical Workflow Automation", "Healthcare Analytics"],
                "budget_allocation": "35% R&D, 30% Sales & Marketing, 25% Operations, 10% Other",
                "match_score": 80,
                "reason": "Excellent match for healthcare tech solutions",
                "articles": [
                    {
                        "title": "HealthSync Partners with Major Hospital Networks",
                        "source": "Healthcare IT News",
                        "date": "January 15, 2025",
                        "url": "https://healthcareitnews.com/healthsync-partnerships",
                        "quote": "The administrative burden on healthcare providers is unsustainable. Our platform reduces documentation time by 40%, giving clinicians more time with patients."
                    }
                ],
                "leads": [
                    {
                        "name": "Dr. Lisa Chen",
                        "title": "Chief Medical Officer",
                        "email": "lchen@healthsync.io",
                        "linkedin": "https://linkedin.com/in/drlisachen"
                    },
                    {
                        "name": "Mark Johnson",
                        "title": "VP of Provider Relations",
                        "email": "mjohnson@healthsync.io",
                        "linkedin": "https://linkedin.com/in/markjohnson"
                    }
                ],
                "events": [
                    {
                        "name": "HIMSS Global Health Conference",
                        "date": "April 15-19, 2025",
                        "location": "Orlando, FL",
                        "url": "https://himssconference.org",
                        "description": "Leading healthcare information and technology conference",
                        "attending_companies": ["HealthSync", "Epic", "Cerner", "Philips"]
                    }
                ]
            },
            {
                "name": "LogisticsAI",
                "website": "https://logisticsai.com",
                "industry": "Supply Chain Technology",
                "size": "100-200 employees",
                "description": "AI-powered supply chain optimization platform for manufacturing and distribution companies.",
                "investment_areas": ["Predictive Logistics", "Inventory Optimization", "Sustainable Supply Chain"],
                "budget_allocation": "40% R&D, 30% Sales & Marketing, 20% Operations, 10% Other",
                "match_score": 78,
                "reason": "Good fit for supply chain technology solutions",
                "articles": [
                    {
                        "title": "LogisticsAI Helps Companies Navigate Supply Chain Disruptions",
                        "source": "Supply Chain Dive",
                        "date": "February 5, 2025",
                        "url": "https://supplychaindive.com/logisticsai-disruptions",
                        "quote": "Our predictive models identified potential disruptions months before they happened, allowing our clients to secure alternative suppliers ahead of the competition."
                    }
                ],
                "leads": [
                    {
                        "name": "Carlos Martinez",
                        "title": "Chief Operations Officer",
                        "email": "cmartinez@logisticsai.com",
                        "linkedin": "https://linkedin.com/in/carlosmartinez"
                    },
                    {
                        "name": "Sophia Lee",
                        "title": "VP of Customer Success",
                        "email": "slee@logisticsai.com",
                        "linkedin": "https://linkedin.com/in/sophialee"
                    }
                ],
                "events": [
                    {
                        "name": "Supply Chain Innovation Summit",
                        "date": "May 10-12, 2025",
                        "location": "Chicago, IL",
                        "url": "https://supplychaininnovation.com",
                        "description": "Conference focused on supply chain technology and innovation",
                        "attending_companies": ["LogisticsAI", "UPS", "FedEx", "DHL"]
                    }
                ]
            }
        ]
        
        # If market or product info is available, try to customize the mock recommendations
        market = self._get_flow_controller_attribute('market', '').lower()
        product = self._get_flow_controller_attribute('product', '').lower()
        
        # Filter or prioritize companies based on market if specified
        if market:
            # Check for industry matches
            industry_map = {
                'tech': ['Enterprise Software', 'Cybersecurity'],
                'healthcare': ['Healthcare Technology'],
                'finance': ['Financial Technology'],
                'manufacturing': ['Supply Chain Technology'],
                'energy': ['CleanTech'],
                'retail': ['Enterprise Software', 'Supply Chain Technology'],
                'education': ['Enterprise Software']
            }
            
            # Find matching industries for the specified market
            matching_industries = []
            for key, industries in industry_map.items():
                if key in market:
                    matching_industries.extend(industries)
            
            # If we found matching industries, filter the recommendations
            if matching_industries:
                # Sort companies by relevance to the market
                companies.sort(key=lambda x: (
                    # Companies in matching industries come first
                    x.get('industry', '') not in matching_industries,
                    # Then sort by match score
                    -x.get('match_score', 0)
                ))
        
        # Limit to requested count
        limited_companies = companies[:count]
        
        # Generate new reason fields based on product/market if available
        if product or market:
            for company in limited_companies:
                if product and market:
                    company['reason'] = f"Great fit for {product} in the {market} sector"
                elif product:
                    company['reason'] = f"Potential customer for your {product} solution"
                elif market:
                    company['reason'] = f"Active buyer in the {market} market"
        
        return limited_companies
