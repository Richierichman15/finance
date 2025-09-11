import json
from typing import Dict, List, Any
from datetime import datetime
from .ollama_service import OllamaService

class FinancialAdvisor:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        self.personality = self._get_advisor_personality()
        self.default_portfolio = self._get_default_portfolio_allocation()
        
    def _get_advisor_personality(self) -> str:
        return """
        You are THE ELITE FINANCIAL ADVISOR - the absolute best in the game! 🚀💰
        
        YOUR PERSONALITY:
        - You're OBSESSED with making insane profits and beating the market
        - You see opportunities where others see risks
        - You're aggressive but smart - calculated risks for maximum gains
        - You speak with confidence and use profit-focused language
        - You're always looking for the next big play
        - You believe small money can become BIG money with the right strategy
        - You're not afraid of volatility - you THRIVE on it
        
        YOUR EXPERTISE:
        - Master of aggressive growth strategies
        - Expert in ETF optimization and sector rotation
        - Specialist in finding undervalued opportunities
        - Timing market entries and exits for maximum profit
        - Building wealth from small amounts through compound strategies
        
        YOUR TONE:
        - Confident and assertive
        - Uses profit-focused language ("Let's make BANK!", "Time to CASH IN!")
        - Direct and action-oriented
        - Slightly aggressive but professional
        - Always optimistic about profit potential
        
        REMEMBER: We're here to make SERIOUS MONEY! Every recommendation should be about maximizing profits while managing risk intelligently.
        """
    
    def _get_default_portfolio_allocation(self) -> Dict[str, Any]:
        return {
            "allocations": [
                {
                    "name": "Broad U.S. Equities",
                    "percentage": 40,
                    "etfs": ["VTI", "SCHB"],
                    "strategy": "Core growth foundation - steady gains with upside potential"
                },
                {
                    "name": "U.S. Energy Sector", 
                    "percentage": 20,
                    "etfs": ["XLE", "AMLP"],
                    "strategy": "Energy play - capitalizing on oil/gas demand and infrastructure"
                },
                {
                    "name": "Global Infrastructure",
                    "percentage": 15, 
                    "etfs": ["IGF", "BIPC"],
                    "strategy": "Real assets play - inflation hedge with growth potential"
                },
                {
                    "name": "Technology Growth",
                    "percentage": 15,
                    "etfs": ["QQQ", "VGT"], 
                    "strategy": "High-growth tech exposure - where the BIG money is made"
                },
                {
                    "name": "Real Estate",
                    "percentage": 5,
                    "etfs": ["VNQ"],
                    "strategy": "REITs for income and diversification"
                },
                {
                    "name": "Speculative Crypto",
                    "percentage": 5,
                    "description": "High-risk, high-reward - the rocket fuel of your portfolio! 🚀"
                }
            ],
            "philosophy": "AGGRESSIVE DIVERSIFIED GROWTH - We're building a money-making MACHINE!"
        }

    async def generate_advice(self, question: str, portfolio_value: float, 
                            risk_tolerance: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized financial advice"""
        
        prompt = f"""
        FINANCIAL SITUATION:
        - Portfolio Value: ${portfolio_value:,.2f}
        - Risk Tolerance: {risk_tolerance}
        - Question: {question}
        
        MARKET CONTEXT:
        - Overall Sentiment: {market_context.get('sentiment', {}).get('overall_sentiment', 'neutral')}
        - Sentiment Details: {json.dumps(market_context.get('sentiment', {}), default=str) if market_context.get('sentiment') else 'N/A'}
        - Top Headlines: {[h.get('title') for h in (market_context.get('news') or [])[:5]] if market_context.get('news') else 'No recent news'}
        
        INSTRUCTIONS (STRICT):
        - Do NOT repeat the same point. Each item must be unique.
        - Always tie advice to the provided news headlines and/or market sentiment. Explicitly reference the context (e.g., if housing is hot, mention REITs; if tech is trending, mention AI or semiconductors).
        - Be specific: name sectors/industries/examples (e.g., cloud computing, biotech, renewable energy) instead of generic labels.
        - Include a timeframe for each item: short-term (1–3 months), mid-term (6–12 months), or long-term (1–3 years).
        - Balance risk with action: include a clear risk management step (stop-loss, position sizing, diversification) for each item.
        - Structure the output as 5–6 bullet items maximum. Each item must include: Opportunity, Reasoning (tied to news/sentiment), Suggested timeframe, Risk management.
        - Tone: elite, confident, precise, and action-oriented. Avoid filler.
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "advice": [
                    {
                        "opportunity": "string",
                        "reasoning": "ties to specific headlines/sentiment",
                        "timeframe": "short-term | mid-term | long-term",
                        "risk_management": "specific action (stop-loss %, position size %, diversification guideline)"
                    }
                ],
                "confidence": "1-100",
                "summary": "one-paragraph elite summary tying items to context"
            }
        )
        
        # Ensure we have fallback values
        if response.get("structured", True):
            items = response.get("advice") or []
            # Keep max 6 unique items
            unique_seen = set()
            deduped = []
            for it in items:
                key = (it.get("opportunity", "").strip().lower(), it.get("timeframe", "").strip().lower())
                if key not in unique_seen and it.get("opportunity"):
                    unique_seen.add(key)
                    deduped.append(it)
                if len(deduped) >= 6:
                    break
            # Convert structured items into string action items for API schema compatibility
            action_strings = []
            for it in deduped:
                opp = it.get("opportunity", "").strip()
                rea = it.get("reasoning", "").strip()
                tfr = it.get("timeframe", "").strip()
                risk = it.get("risk_management", "").strip()
                if opp:
                    action_strings.append(f"Opportunity: {opp} | Reason: {rea} | Timeframe: {tfr} | Risk: {risk}")
            return {
                "response": response.get("summary") or "Elite plan prepared. See structured items.",
                "confidence": response.get("confidence", 85),
                "actions": action_strings,
                "profit_timeline": "Item-specific (see timeframes)"
            }
        else:
            return {
                "response": response.get("response", "Elite plan: pursue 5–6 unique, context-tied opportunities with defined timeframes and risk controls."),
                "confidence": 80,
                "actions": [
                    "Opportunity: Momentum in AI/semiconductors via QQQ/VGT | Reason: Tech leadership per sentiment/news | Timeframe: short-term | Risk: 5–8% stop-loss; max 10% position"
                ],
                "profit_timeline": "Short to medium term"
            }

    async def get_portfolio_advice(self, portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific portfolio optimization advice"""
        
        prompt = f"""
        PORTFOLIO ANALYSIS:
        {json.dumps(portfolio_analysis, indent=2)}
        
        Produce AGGRESSIVE optimization recommendations following these strict rules:
        - Do NOT repeat the same point. Each item must be unique.
        - Tie each recommendation to current market context if provided (news/sentiment) and/or metrics in the analysis.
        - Be specific (name industries/sectors/examples).
        - Include a timeframe per item: short-term (1–3m), mid-term (6–12m), long-term (1–3y).
        - Include a concrete risk management step per item (stop-loss %, position sizing %, diversification rule).
        - Output 5–6 bullet items MAX, each with: Opportunity, Reasoning, Suggested timeframe, Risk management.
        - Tone: elite, confident, precise.
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "recommendations": [
                    {
                        "opportunity": "string",
                        "reasoning": "ties to portfolio/market context",
                        "timeframe": "short-term | mid-term | long-term",
                        "risk_management": "specific control"
                    }
                ],
                "risk_assessment": "risk analysis with profit focus",
                "profit_potential": "potential gains assessment",
                "immediate_actions": ["urgent moves to make"]
            }
        )
        
        if response.get("structured", True):
            # Deduplicate and cap at 6 items
            recs = response.get("recommendations") or []
            unique = []
            seen = set()
            for r in recs:
                key = (r.get("opportunity", "").strip().lower(), r.get("timeframe", "").strip().lower())
                if key not in seen and r.get("opportunity"):
                    seen.add(key)
                    unique.append(r)
                if len(unique) >= 6:
                    break
            # Convert to string recommendations for API schema
            rec_strings = []
            for r in unique:
                opp = r.get("opportunity", "").strip()
                rea = r.get("reasoning", "").strip()
                tfr = r.get("timeframe", "").strip()
                risk = r.get("risk_management", "").strip()
                if opp:
                    rec_strings.append(f"Opportunity: {opp} | Reason: {rea} | Timeframe: {tfr} | Risk: {risk}")
            response["recommendations"] = rec_strings
            return response
        else:
            return {
                "recommendations": [
                    "Opportunity: Rebalance toward tech momentum (AI/cloud/semis) | Reason: Leadership in growth sectors per analysis | Timeframe: short-term | Risk: 5% stop; 10% position cap"
                ],
                "risk_assessment": "Calculated risks aligned with aggressive growth",
                "profit_potential": "High with disciplined execution",
                "immediate_actions": ["Implement rebalancing", "Set risk controls"]
            }

    async def comment_on_rebalancing(self, rebalance_plan: Dict[str, Any]) -> str:
        """Provide AI commentary on rebalancing recommendations"""
        
        prompt = f"""
        REBALANCING PLAN:
        {json.dumps(rebalance_plan, indent=2)}
        
        Provide your expert commentary on this rebalancing plan:
        - Is this aggressive enough for maximum profits?
        - Any tweaks to squeeze out more gains?
        - Market timing considerations
        - Expected profit impact
        
        Keep it confident and profit-focused!
        """
        
        response = await self.ollama.generate_response(
            prompt=prompt,
            system_prompt=self.personality
        )
        
        return response

    async def analyze_market_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific market opportunity"""
        
        prompt = f"""
        MARKET OPPORTUNITY:
        {json.dumps(opportunity_data, indent=2)}
        
        Analyze this opportunity like the profit-focused advisor you are:
        1. Profit potential rating (1-10)
        2. Risk vs reward analysis
        3. Entry strategy
        4. Exit strategy
        5. Position sizing recommendation
        6. Timeline for maximum gains
        
        Should we GO ALL IN or pass on this one?
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "profit_rating": "1-10 rating",
                "analysis": "detailed opportunity analysis",
                "entry_strategy": "how to enter position",
                "position_size": "recommended allocation",
                "go_no_go": "PURSUE or PASS"
            }
        )
        
        return response

    def get_aggressive_strategies(self) -> List[Dict[str, Any]]:
        """Get list of aggressive investment strategies"""
        return [
            {
                "name": "Sector Rotation Play",
                "description": "Rotate into highest momentum sectors quarterly",
                "risk_level": 7,
                "profit_potential": "15-25% annually",
                "execution": "Use sector ETFs like XLE, XLF, XLK"
            },
            {
                "name": "Growth Concentration",
                "description": "Overweight high-growth tech and innovation",
                "risk_level": 8,
                "profit_potential": "20-40% annually",
                "execution": "QQQ, ARKK, VGT heavy allocation"
            },
            {
                "name": "Energy Infrastructure Boom",
                "description": "Capitalize on energy transition infrastructure",
                "risk_level": 6,
                "profit_potential": "12-20% annually", 
                "execution": "AMLP, KMI, ENB, infrastructure plays"
            },
            {
                "name": "Crypto Allocation",
                "description": "5-10% in crypto for explosive upside",
                "risk_level": 10,
                "profit_potential": "50-200% potential",
                "execution": "BTC, ETH via regulated exchanges"
            }
        ]

    def get_market_timing_signals(self) -> Dict[str, str]:
        """Get current market timing signals"""
        return {
            "overall_market": "BULLISH - Fed pivot creates opportunity",
            "energy_sector": "STRONG BUY - Supply constraints + demand growth",
            "tech_growth": "ACCUMULATE - AI revolution just beginning",
            "real_estate": "SELECTIVE - Focus on data centers and logistics",
            "crypto": "VOLATILE - DCA strategy for long-term gains"
        }