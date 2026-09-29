from agents.specialists.transaction import TransactionAgent
from agents.specialists.engagement import EngagementAgent
from agents.specialists.support import SupportAgent
from agents.specialists.investment import InvestmentAgent
from agents.specialists.compliance import ComplianceAgent


def build_agents(llm=None) -> dict:
    agents = [TransactionAgent(llm), EngagementAgent(llm), SupportAgent(llm),
              InvestmentAgent(llm), ComplianceAgent(llm)]
    return {a.key: a for a in agents}
