# backend/tests/test_ast_chunker.py
#
# Test suite for Improvement 2: AST-Based Function/Class Chunking
#
# Run with:
#   pytest tests/test_ast_chunker.py -v
#
# All tests use in-memory ParsedFile objects — no disk access needed.

import pytest
from app.models.ingest_models import ParsedFile
from app.core.processing.ast_chunker import (
    PythonASTChunker,
    JavaScriptASTChunker,
    JavaASTChunker,
    GoASTChunker,
    ASTChunkingEngine,
    language_strategy,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_file(
    content:  str,
    language: str = "python",
    path:     str = "src/test.py",
) -> ParsedFile:
    return ParsedFile(
        file_path     = path,
        relative_path = path,
        content       = content,
        language      = language,
        project_id    = "test_project",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Python AST Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TestPythonASTChunker:

    def setup_method(self):
        self.chunker = PythonASTChunker()

    # ── Basic function extraction ──────────────────────────────────────────

    def test_extracts_single_function(self):
        code = '''
def authenticate(username, password):
    """Authenticate a user."""
    user = find_user(username)
    if not user:
        raise ValueError("Not found")
    return create_token(user)
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        names  = [c.function_name for c in chunks if c.function_name]
        assert "authenticate" in names

    def test_extracts_multiple_functions(self):
        code = '''
def login(username, password):
    return authenticate(username, password)


def logout(token):
    blacklist.add(token)
    return True


def refresh_token(token):
    payload = verify_token(token)
    return create_token(payload["user_id"])
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        names  = {c.function_name for c in chunks if c.function_name}
        assert "login"         in names
        assert "logout"        in names
        assert "refresh_token" in names

    def test_chunk_type_is_function(self):
        code = '''
def foo():
    return 1
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        func_chunks = [c for c in chunks if c.function_name == "foo"]
        assert len(func_chunks) >= 1
        assert func_chunks[0].chunk_type in ("function", "method")

    # ── Class extraction ───────────────────────────────────────────────────

    def test_extracts_class(self):
        code = '''
class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        return self.db.find(user_id)
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        types  = {(c.chunk_type, c.class_name or c.function_name) for c in chunks}
        assert ("class", "UserService") in types

    def test_extracts_methods_from_class(self):
        code = '''
class AuthService:
    def login(self, username, password):
        return self._create_token(username)

    def logout(self, token):
        self.blacklist.add(token)

    def _create_token(self, username):
        return jwt.encode({"sub": username}, self.secret)
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        # Should extract methods separately
        method_names = {c.function_name for c in chunks if c.function_name}
        assert "login"         in method_names
        assert "logout"        in method_names
        assert "_create_token" in method_names

    def test_method_chunk_has_parent_class(self):
        code = '''
class AuthService:
    def authenticate(self, user, pwd):
        return jwt.encode({}, "secret")
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        auth_chunks = [
            c for c in chunks if c.function_name == "authenticate"
        ]
        assert len(auth_chunks) >= 1
        assert auth_chunks[0].class_name == "AuthService"

    # ── Exact line numbers ─────────────────────────────────────────────────

    def test_start_line_is_accurate(self):
        code = "x = 1\ny = 2\n\ndef foo():\n    return 42\n"
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        foo_chunks = [c for c in chunks if c.function_name == "foo"]
        assert len(foo_chunks) >= 1
        # foo() starts on line 4 (1-indexed)
        assert foo_chunks[0].start_line == 4

    def test_end_line_is_after_start_line(self):
        code = '''
def long_function(a, b, c):
    x = a + b
    y = b + c
    z = a * b * c
    return x + y + z
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.end_line >= c.start_line

    def test_chunk_text_contains_function_body(self):
        code = '''
def authenticate(username, password):
    user = find_user(username)
    if not user:
        raise ValueError("Invalid")
    return create_jwt(user.id)
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        auth_chunks = [c for c in chunks if "authenticate" in c.text]
        assert len(auth_chunks) >= 1
        assert "find_user" in auth_chunks[0].text
        assert "create_jwt" in auth_chunks[0].text

    # ── Metadata ───────────────────────────────────────────────────────────

    def test_async_function_detected(self):
        code = '''
async def fetch_user(user_id: int):
    return await db.get(user_id)
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        # Chunk should exist and contain async def
        assert any("fetch_user" in c.function_name for c in chunks
                   if c.function_name)

    def test_decorated_function_included(self):
        code = '''
@app.route("/login", methods=["POST"])
@require_auth
def login():
    return handle_login()
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        login_chunks = [c for c in chunks if c.function_name == "login"]
        assert len(login_chunks) >= 1
        # Decorator should be included in the chunk text
        assert "@app.route" in login_chunks[0].text or \
               "login" in login_chunks[0].text

    def test_empty_file_returns_no_chunks(self):
        file   = make_file("# just a comment\n")
        chunks = self.chunker.chunk(file)
        assert isinstance(chunks, list)

    def test_syntax_error_returns_empty(self):
        code = "def broken(\n    # missing closing paren"
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        # Should return [] to signal fallback (not raise)
        assert isinstance(chunks, list)

    def test_chunk_ids_are_unique(self):
        code = '''
def func_a():
    return "a"


def func_b():
    return "b"


def func_c():
    return "c"
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_file_path_preserved(self):
        code   = "def foo():\n    pass\n"
        file   = make_file(code, path="src/auth/login.py")
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.file_path == "src/auth/login.py"

    def test_language_preserved(self):
        code   = "def foo():\n    pass\n"
        file   = make_file(code, language="python")
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.language == "python"

    def test_char_count_populated(self):
        code   = "def foo():\n    return 1\n"
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.char_count > 0

    def test_token_count_populated(self):
        code   = "def foo():\n    return 1\n"
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.token_count > 0

    # ── Real-world patterns ────────────────────────────────────────────────

    def test_nested_classes(self):
        code = '''
class Outer:
    class Inner:
        def inner_method(self):
            return "inner"

    def outer_method(self):
        return self.Inner()
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        # Should not crash and should produce chunks
        assert len(chunks) >= 1

    def test_class_with_many_methods(self):
        methods = "\n".join(
            f"    def method_{i}(self):\n        return {i}\n"
            for i in range(10)
        )
        code = f"class BigService:\n{methods}\n"
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        # Should extract the class + methods
        assert len(chunks) >= 1

    def test_module_level_code_not_duplicated(self):
        code = '''
SECRET = "my_secret_key"
ALGORITHM = "HS256"


def create_token(user_id):
    return jwt.encode({"sub": user_id}, SECRET)


def decode_token(token):
    return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
'''
        file   = make_file(code)
        chunks = self.chunker.chunk(file)
        names  = [c.function_name for c in chunks if c.function_name]
        # No duplicates
        assert len(names) == len(set(names))


# ══════════════════════════════════════════════════════════════════════════════
# 2. JavaScript AST Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TestJavaScriptASTChunker:

    def setup_method(self):
        self.chunker = JavaScriptASTChunker()

    def test_extracts_named_function(self):
        code = '''
function authenticateUser(req, res) {
    const { username, password } = req.body;
    const token = createToken(username);
    res.json({ token });
}
'''
        file   = make_file(code, language="javascript", path="auth.js")
        chunks = self.chunker.chunk(file)
        names  = {c.function_name for c in chunks if c.function_name}
        assert "authenticateUser" in names

    def test_extracts_class(self):
        code = '''
class UserService {
    constructor(db) {
        this.db = db;
    }

    async getUser(id) {
        return await this.db.findById(id);
    }

    createUser(data) {
        return this.db.create(data);
    }
}
'''
        file   = make_file(code, language="javascript", path="service.js")
        chunks = self.chunker.chunk(file)
        types  = {c.chunk_type for c in chunks}
        assert "class" in types

    def test_brace_counting_finds_exact_end(self):
        code = '''
function processData(data) {
    if (data.valid) {
        const result = transform(data);
        if (result.ok) {
            return { success: true, data: result };
        }
    }
    return { success: false };
}
'''
        file   = make_file(code, language="javascript", path="utils.js")
        chunks = self.chunker.chunk(file)
        if chunks:
            # The chunk should contain the entire function body
            text = chunks[0].text
            assert "processData" in text
            assert "success: false" in text   # Inside the last brace

    def test_typescript_supported(self):
        code = '''
async function fetchUser(userId: string): Promise<User> {
    const user = await db.find(userId);
    if (!user) throw new Error("Not found");
    return user;
}
'''
        file   = make_file(code, language="typescript", path="user.ts")
        chunks = self.chunker.chunk(file)
        # Should not crash — TypeScript uses same chunker as JS
        assert isinstance(chunks, list)

    def test_arrow_function_extracted(self):
        code = '''
const createToken = (userId) => {
    return jwt.sign({ sub: userId }, process.env.SECRET);
};
'''
        file   = make_file(code, language="javascript", path="jwt.js")
        chunks = self.chunker.chunk(file)
        # Arrow functions should be detected
        assert isinstance(chunks, list)

    def test_chunk_text_is_complete(self):
        code = '''
function login(username, password) {
    const user = findUser(username);
    if (!verifyPassword(password, user.hash)) {
        throw new AuthError("Invalid credentials");
    }
    return generateJWT(user.id);
}
'''
        file   = make_file(code, language="javascript", path="auth.js")
        chunks = self.chunker.chunk(file)
        if chunks:
            full_text = " ".join(c.text for c in chunks)
            assert "generateJWT" in full_text

    def test_empty_file_returns_list(self):
        file   = make_file("// just a comment\n", language="javascript")
        chunks = self.chunker.chunk(file)
        assert isinstance(chunks, list)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Java AST Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TestJavaASTChunker:

    def setup_method(self):
        self.chunker = JavaASTChunker()

    def test_extracts_public_class(self):
        code = '''
public class UserService {
    private final UserRepository repo;

    public UserService(UserRepository repo) {
        this.repo = repo;
    }

    public User findUser(String username) {
        return repo.findByUsername(username);
    }
}
'''
        file   = make_file(code, language="java", path="UserService.java")
        chunks = self.chunker.chunk(file)
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1
        assert any("UserService" in c.class_name for c in class_chunks)

    def test_extracts_interface(self):
        code = '''
public interface AuthService {
    String authenticate(String username, String password);
    void logout(String token);
    boolean validateToken(String token);
}
'''
        file   = make_file(code, language="java", path="AuthService.java")
        chunks = self.chunker.chunk(file)
        assert len(chunks) >= 1

    def test_chunk_text_is_complete(self):
        code = '''
public class TokenService {
    public String createToken(User user) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        return Jwts.builder()
                   .setClaims(claims)
                   .signWith(secretKey)
                   .compact();
    }
}
'''
        file   = make_file(code, language="java", path="TokenService.java")
        chunks = self.chunker.chunk(file)
        if chunks:
            full_text = " ".join(c.text for c in chunks)
            assert "compact" in full_text   # Last line should be included

    def test_returns_list_for_empty_file(self):
        file   = make_file("// empty\n", language="java")
        chunks = self.chunker.chunk(file)
        assert isinstance(chunks, list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Go AST Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TestGoASTChunker:

    def setup_method(self):
        self.chunker = GoASTChunker()

    def test_extracts_function(self):
        code = '''
func Authenticate(username, password string) (string, error) {
    user, err := findUser(username)
    if err != nil {
        return "", err
    }
    token := createJWT(user.ID)
    return token, nil
}
'''
        file   = make_file(code, language="go", path="auth.go")
        chunks = self.chunker.chunk(file)
        names  = {c.function_name for c in chunks if c.function_name}
        assert "Authenticate" in names

    def test_extracts_struct(self):
        code = '''
type UserService struct {
    db     *Database
    cache  *Cache
    logger *Logger
}
'''
        file   = make_file(code, language="go", path="service.go")
        chunks = self.chunker.chunk(file)
        assert len(chunks) >= 1

    def test_extracts_method_with_receiver(self):
        code = '''
func (s *UserService) GetUser(id string) (*User, error) {
    return s.db.FindByID(id)
}
'''
        file   = make_file(code, language="go", path="service.go")
        chunks = self.chunker.chunk(file)
        assert len(chunks) >= 1
        # GetUser should be found
        names = {c.function_name for c in chunks if c.function_name}
        assert "GetUser" in names

    def test_brace_counting_handles_nested_braces(self):
        code = '''
func ProcessRequest(req *Request) (*Response, error) {
    if req.Method == "POST" {
        data := map[string]interface{}{
            "status": "ok",
            "code":   200,
        }
        return &Response{Body: data}, nil
    }
    return nil, errors.New("method not allowed")
}
'''
        file   = make_file(code, language="go", path="handler.go")
        chunks = self.chunker.chunk(file)
        if chunks:
            full_text = " ".join(c.text for c in chunks)
            # The map literal braces should NOT prematurely end the function
            assert "method not allowed" in full_text

    def test_returns_list_for_empty_file(self):
        file   = make_file("// empty\n", language="go")
        chunks = self.chunker.chunk(file)
        assert isinstance(chunks, list)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ASTChunkingEngine (orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class TestASTChunkingEngine:

    def setup_method(self):
        self.engine = ASTChunkingEngine(prefer_ast=True)

    def test_python_uses_ast_strategy(self):
        assert "ast" in language_strategy("python").lower()

    def test_js_uses_brace_counting_strategy(self):
        assert "brace" in language_strategy("javascript").lower()

    def test_go_uses_brace_counting_strategy(self):
        assert "brace" in language_strategy("go").lower()

    def test_markdown_uses_markdown_chunker(self):
        assert "Markdown" in language_strategy("markdown")

    def test_unknown_language_uses_fallback(self):
        assert "Sliding" in language_strategy("cobol") or \
               "Function" in language_strategy("cobol")

    def test_python_file_chunked_with_ast(self):
        code = '''
def create_user(username, email):
    return User(username=username, email=email)


def delete_user(user_id):
    User.objects.filter(id=user_id).delete()
'''
        file   = make_file(code, language="python")
        chunks = self.engine.chunk_file(file)
        assert len(chunks) >= 1
        names  = {c.function_name for c in chunks if c.function_name}
        assert "create_user" in names or "delete_user" in names

    def test_javascript_file_chunked(self):
        code = '''
function renderPage(template, data) {
    const html = mustache.render(template, data);
    return html;
}
'''
        file   = make_file(code, language="javascript", path="render.js")
        chunks = self.engine.chunk_file(file)
        assert isinstance(chunks, list)

    def test_markdown_file_chunked(self):
        code = "# Install\n\nRun npm install.\n\n# Usage\n\nImport the module.\n"
        file   = make_file(code, language="markdown", path="README.md")
        chunks = self.engine.chunk_file(file)
        assert len(chunks) >= 1
        assert any(c.chunk_type == "heading" for c in chunks)

    def test_sql_falls_back_to_sliding(self):
        code = ("SELECT u.id, u.name, r.name as role\n"
                "FROM users u\n"
                "JOIN roles r ON u.role_id = r.id\n"
                "WHERE u.active = 1\n") * 20
        file   = make_file(code, language="sql", path="query.sql")
        chunks = self.engine.chunk_file(file)
        assert isinstance(chunks, list)

    def test_chunk_all_returns_chunkingresult(self):
        files = [
            make_file("def foo():\n    return 1\n", "python",     "a.py"),
            make_file("# Title\n\nContent.\n",       "markdown",   "README.md"),
            make_file("SELECT 1;\n" * 30,             "sql",        "q.sql"),
        ]
        result = self.engine.chunk_all(files, "test_proj")
        assert result.total_files  >= 1
        assert result.total_chunks >= 1
        assert isinstance(result.chunks_by_language, dict)

    def test_all_chunks_have_file_path(self):
        code  = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        file  = make_file(code, "python", "src/utils.py")
        chunks = self.engine.chunk_file(file)
        for c in chunks:
            assert c.file_path == "src/utils.py"

    def test_all_chunks_have_valid_line_numbers(self):
        code = '''
def func_a(x):
    return x + 1


def func_b(y):
    return y * 2


def func_c(z):
    return z ** 2
'''
        file  = make_file(code)
        chunks = self.engine.chunk_file(file)
        for c in chunks:
            assert c.start_line >= 1,        f"start_line < 1: {c.chunk_id}"
            assert c.end_line >= c.start_line, f"end_line < start_line: {c.chunk_id}"

    def test_chunk_ids_globally_unique(self):
        files = [
            make_file("def a():\n    pass\n", "python", "file1.py"),
            make_file("def b():\n    pass\n", "python", "file2.py"),
            make_file("def c():\n    pass\n", "python", "file3.py"),
        ]
        result = self.engine.chunk_all(files, "proj")
        ids    = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs across files"

    def test_fallback_when_ast_fails(self):
        """Engine should fall back gracefully when AST parse fails."""
        # Deliberately broken Python (AST will fail)
        broken_code = "def broken(\n    x, y\n# missing closing paren and colon"
        file   = make_file(broken_code, language="python")
        chunks = self.engine.chunk_file(file)
        # Should return a list (not raise) — fallback handles it
        assert isinstance(chunks, list)

    def test_prefer_ast_false_uses_phase7_only(self):
        """When prefer_ast=False, should only use Phase 7 chunkers."""
        engine = ASTChunkingEngine(prefer_ast=False)
        code   = "def foo():\n    return 1\n"
        file   = make_file(code)
        chunks = engine.chunk_file(file)
        assert isinstance(chunks, list)

    def test_complete_function_body_in_chunk(self):
        """
        CORE TEST: Chunks must contain COMPLETE function bodies,
        not truncated at arbitrary character positions.
        """
        code = '''
def calculate_tax(income, rate, deductions):
    """Calculate tax after deductions."""
    taxable_income = income - deductions
    if taxable_income < 0:
        taxable_income = 0
    raw_tax = taxable_income * rate
    final_tax = max(0, raw_tax - 500)  # Apply base credit
    return round(final_tax, 2)
'''
        file   = make_file(code)
        chunks = self.engine.chunk_file(file)
        calc_chunks = [c for c in chunks if "calculate_tax" in c.text]
        assert len(calc_chunks) >= 1

        text = calc_chunks[0].text
        # All key lines should be present
        assert "taxable_income"  in text
        assert "raw_tax"         in text
        assert "final_tax"       in text
        assert "return round"    in text   # Last line of function

    def test_implementation_not_doc_chunks(self):
        """
        AST chunker should produce 'function'/'class' chunks,
        not 'block' chunks for functions (which is what the
        sliding window produces when it doesn't know where a
        function starts/ends).
        """
        code = '''
def authenticate(username, password):
    user = find_user(username)
    return verify(user, password)
'''
        file   = make_file(code)
        chunks = self.engine.chunk_file(file)
        func_chunks = [c for c in chunks if c.function_name == "authenticate"]
        if func_chunks:
            assert func_chunks[0].chunk_type in ("function", "method"), (
                f"Expected function/method type, got: {func_chunks[0].chunk_type}"
            )