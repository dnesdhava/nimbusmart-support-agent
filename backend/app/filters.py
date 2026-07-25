from semantic_kernel import Kernel
from semantic_kernel.filters import FilterTypes
from semantic_kernel.functions import FunctionResult
from opentelemetry import trace
from app.auth import STAFF_ROLES
from app.orders_data import ORDERS
from app.session import ChatSession

_tracer = trace.get_tracer("nimbus.guardrail")

BLOCKED_KEYWORDS = ("card number", "cvv", "credit_number", "creditcard")

def _block(context, message: str) -> None:
    context.result = FunctionResult(function=context.function.metadata, value=message)

def register_guardrail_filter(kernel: Kernel, session: ChatSession) -> None:
    @kernel.filter(FilterTypes.FUNCTION_INVOCATION)
    async def gaurd(context, next):
        args = {k: str(v) for k, v in dict(context.arguments).items()}
        event = {
            "plugin": context.function.plugin_name,
            "function": context.function.name,
            "args": args,
            "blocked": False,
        }
        with _tracer.start_as_current_span("guardrail.check") as span:
            span.set_attribute("tool.plugin", context.function.plugin_name)
            span.set_attribute("tool.function", context.function.name)
            span.set_attribute("user.role", getattr(session.user, "role", "anonymous"))
            if any(keyword in str(args).lower() for keyword in BLOCKED_KEYWORDS):
                _block(context, "Blocked due to sensitive information in arguments.")
                event["blocked"] = True
                span.set_attribute("decision", "blocked_card")
                session.tool_calls.append(event)
                return
            if context.function.plugin_name == "Orders" and not _can_view_order(session, args.get("order_id", "")):
                _block(context, "Blocked due to insufficient permissions to view order.")
                event["blocked"] = True
                span.set_attribute("decision", "denied_rbac")
                session.tool_calls.append(event)
                return
            span.set_attribute("decision", "allow")
            try:
                await next(context)
            finally:
                session.tool_calls.append(event)

def _can_view_order(session: ChatSession, order_id: str) -> bool:
    user = session.user
    if user is None: 
        return False
    if user.role in STAFF_ROLES:
        return True
    order = ORDERS.get(order_id.upper())
    return bool(order) and order.get("customer_id") == user.customer_id