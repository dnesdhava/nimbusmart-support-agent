import json
import secrets # creating secure tokens
from dataclasses import dataclass
from app.config import USERS_PATH

STAFF_ROLES = ["admin", "manager", "salesperson"]

_USERS: dict = json.loads(USERS_PATH.read_text(encoding="utf-8"))
_TOKENS: dict[str, "User"] = {}

@dataclass(frozen=True)
class User:
    username: str
    role: str
    customer_id: str | None
    display_name: str

    @property
    def is_staff(self) -> bool:
        return self.role in STAFF_ROLES
    
def login(username: str, password: str) -> str | None:
    record = _USERS.get(username)
    if not record or record["password"] != password:
        return None
    user = User(
        username=username,
        role=record["role"],
        customer_id=record["customer_id"],
        display_name=record["display_name"],
    )
    token = secrets.token_urlsafe(24)
    _TOKENS[token] = user
    return token

def get_user(token: str) -> User | None:
    return _TOKENS.get(token)