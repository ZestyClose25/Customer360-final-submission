MAX_MANAGER_ROUNDS = 3 

SPECIALISTS = ["transaction", "engagement", "support", "investment", "compliance"]

AGENT_SOURCE_ACCESS = {
    "transaction": ["card_payments", "instant_payments", "ach_wire", "core_banking_ledger"],
    "engagement":  ["web_app_events"],
    "support":     ["support_logs", "social_signal_consented"],
    "investment":  ["trading_brokerage"],
    "compliance":  ["card_payments", "instant_payments", "ach_wire", "core_banking_ledger", "loan_kyc"],
}

SOURCE_TO_AGENTS = {
    "card_payments": ["transaction", "compliance"],
    "instant_payments": ["transaction", "compliance"],
    "ach_wire": ["transaction", "compliance"],
    "core_banking_ledger": ["transaction"],
    "trading_brokerage": ["investment", "compliance"],
    "loan_kyc": ["compliance"],
    "web_app_events": ["engagement"],
    "support_logs": ["support"],
    "social_signal_consented": ["support"],
}

ALLOWED_HANDOFFS = {
    "compliance": ["transaction", "investment"],
    "transaction": ["compliance"],
    "support": ["transaction", "compliance"],
    "investment": ["compliance"],
    "engagement": ["support"],
}