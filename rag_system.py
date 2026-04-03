"""
=================================================================================
UNIFIED MULTI-DATABASE RAG SYSTEM (OPTIMIZED + WIKIDATA SPARQL)
=================================================================================
Integrates MySQL, FAISS, and Wikibase (Internal) + Wikidata (External).
Features:
- LLM-based SPARQL generation for both Internal Wikibase and External Wikidata.
- Optimized routing (skips internal planning for purely external questions).
- Robust User-Agent headers for Wikidata API.
=================================================================================
"""

import os
import json
import faiss
import numpy as np
import mysql.connector
import requests
from datetime import datetime, date
from openai import OpenAI
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pymongo import MongoClient
import os
 #=================================================================================
# CONFIGURATION  — all values come from environment variables
# =================================================================================
 
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
SERPAPI_KEY     = os.environ["SERPAPI_KEY"]
 
DB_CONFIG = {
    "host":     os.environ["MYSQL_HOST"],
    "port":     int(os.getenv("MYSQL_PORT", "3307")),
    "user":     os.environ["MYSQL_USER"],
    "password": os.environ["MYSQL_PASSWORD"],
    "database": os.environ["MYSQL_DATABASE"],
}
 
# FAISS files must be committed to the repo root (or a /data sub-folder).
# Default paths assume they sit at the repo root alongside this file.
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "wikidata_faiss.index")
METADATA_PATH    = os.getenv("METADATA_PATH",    "faiss_document.json")
 
WIKIBASE_CONFIG = {
    "api_url":   os.environ["WIKIBASE_API_URL"],
    "sparql_url": os.environ["WIKIBASE_SPARQL_URL"],
    "username":  os.environ["WIKIBASE_USERNAME"],
    "password":  os.environ["WIKIBASE_PASSWORD"],
}
 
MONGO_CONFIG = {
    "host":       os.environ["MONGO_HOST"],
    "port":       int(os.getenv("MONGO_PORT", "27017")),
    "username":   os.environ["MONGO_USERNAME"],
    "password":   os.environ["MONGO_PASSWORD"],
    "database":   os.getenv("MONGO_DATABASE", "wikidata"),
    "collection": os.getenv("MONGO_COLLECTION", "news"),
}


MONGODB_PARTNERSHIP_SCHEMA = """
MONGODB PARTNERSHIP COLLECTION — `news`

Each document represents a crawled news/article page about a Telkom University
partnership event (MOU signing, collaboration announcement, etc.).

Fields used by this system:
  - partner_name  (str)  : Full name of the partner institution
                           e.g. "Universitas Borneo Tarakan"
  - title         (str)  : Page / article title
                           e.g. "Penandatanganan MOU Telkom University & ..."
  - summary       (str)  : Short auto-generated summary of the article
  - clean_text    (str)  : Full cleaned article body (used for FAISS content)
  - faiss_id      (int)  : ID into the shared FAISS vector index
  - partner_id    (int)  : Numeric ID for the partner institution

Typical query patterns:
  - Find by partner name  → filter on partner_name  (regex / $regex)
  - Find by keyword       → filter on title or summary ($regex)
  - Find by partner_id    → exact match on partner_id
  - Fetch docs by faiss_ids → filter on faiss_id: {$in: [...]}
  - use faiss for semantic search, then get partner_name for context using fiass_id
"""
# Model Configuration
MODEL_CONTEXT_DETERMINER = "gpt-4.1-mini"
MODEL_QUERY_REASONER = "gpt-4.1-mini"
MODEL_SQL_GENERATOR = "gpt-4.1-mini"
MODEL_SPARQL_GENERATOR = "gpt-4.1-mini" # Used for both Internal and External SPARQL
MODEL_ANSWER_SYNTHESIZER = "gpt-4o-mini"
MODEL_EMBEDDING = "text-embedding-3-small"

FAISS_TOP_K = 3

# Schemas
MYSQL_SCHEMA = """
**CRITICAL** all table names use lowercase letters
always add wikidb. before table name when querying

CREATE TABLE `dosen` (
  `id` VARCHAR(8),
  `nama` VARCHAR(64),
  `namaPT` VARCHAR(64),
  `namaProdi` VARCHAR(64),
  `jenisKelamin` VARCHAR(16),
  `jabatanAkademik` VARCHAR(32),
  `pendidikanTertinggi` VARCHAR(8),
  `statusIkatanKerja` VARCHAR(16),
  `statusAktivitas` VARCHAR(16)['Aktif', 'Tugas Belajar'],
  PRIMARY KEY (`id`)
);

CREATE TABLE `projectassignment` (
  `id` VARCHAR(16),
  `idDosen` VARCHAR(8),
  `projectID` VARCHAR(16),
  `peran` VARCHAR(16),
  PRIMARY KEY (`id`),
  FOREIGN KEY (`idDosen`) REFERENCES `Dosen`(`id`)
);

CREATE TABLE `project` (
  `id` VARCHAR(16),
  `title` VARCHAR(64),
  `budget` Integer,
  PRIMARY KEY (`id`)
);

CREATE TABLE `meetingminutes` (
  `id` VARCHAR(16),
  `faiss_id` VARCHAR(16),
  `date` date,
  `title` VARCHAR(128),
  PRIMARY KEY (`id`)
);

CREATE TABLE `matakuliah` (
  `id` VARCHAR(16),
  `nama` VARCHAR(64),
  `jumlahSKS` int,
  PRIMARY KEY (`id`)
);

CREATE TABLE `employeecontract` (
  `id` VARCHAR(16),
  `IdDosen` VARCHAR(8),
  `faiss_id` VARCHAR(8),
  `contractDate` date,
  `baseSalary` int,
  `signatory` VARCHAR(64),
  PRIMARY KEY (`id`)
);

CREATE TABLE `mengajar` (
  `id` VARCHAR(128),
  `idDosen` VARCHAR(8),
  `MataKuliahID` VARCHAR(16),
  `tahunAkademikStart` INT,
  `tahunAkademikEnd` INT,
  `Semester` VARCHAR(8),
  PRIMARY KEY (`id`),
  FOREIGN KEY (`MataKuliahID`) REFERENCES `MataKuliah`(`id`),
  FOREIGN KEY (`id`) REFERENCES `Dosen`(`id`)
);
"""

WIKIBASE_SCHEMA = """
WIKIBASE KNOWLEDGE GRAPH SCHEMA:

Properties:
- P1 (Has Researched): Lecturer -> wdt:P1 -> Paper
- P2 (Has Patent): Lecturer -> wdt:P2 -> Patent
- P3 (is Lecturer): Entity -> wdt:P3 -> [] (Type check)
- P4 (is Paper): Entity -> wdt:P4 -> [] (Type check)
- P5 (is Patent): Entity -> wdt:P5 -> [] (Type check)
- P6 (Has Partnership): Institution -> wdt:P6 -> Partner
- P7 (Has Faculty): Institution -> wdt:P7 -> Faculty
- P8 (Affiliation): Person -> wdt:P8 -> Organization

SPARQL Prefixes:
PREFIX wd:       <http://38.147.122.59/entity/>
PREFIX wdt:      <http://38.147.122.59/prop/direct/>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""

FAISS_SCHEMA = """
FAISS VECTOR DATABASE:
Contains document content from:
- Meeting minutes (full text, agenda, decisions, action items)
- Employment contracts (terms, conditions, salary details)

Metadata (JSON) includes:
- faiss_id: unique identifier
- document_content: full document text
"""


# =================================================================================
# DATA CLASSES
# =================================================================================


@dataclass
class QueryPlan:
    """Plan for executing queries across databases"""
    needs_mysql: bool
    needs_faiss: bool
    needs_wikibase: bool
    needs_wikidata: bool
    needs_mongodb: bool
    mysql_question: Optional[str]
    faiss_question: Optional[str]
    wikibase_question: Optional[str]
    wikidata_query: Optional[str]
    reasoning: str
    faiss_strategy: Optional[str] = "faiss_direct"  # "sql_first" or "faiss_direct"
    mongodb_question: Optional[str] = None
    mongodb_strategy: Optional[str] = "mongo_first"   # "mongo_first" | "faiss_direct"


@dataclass
class ExecutionStats:
    llm_calls: int = 0
    databases_used: List[str] = field(default_factory=list)
    mysql_queries: List[str] = field(default_factory=list)
    faiss_queries: List[str] = field(default_factory=list)
    wikibase_queries: List[str] = field(default_factory=list)
    wikidata_searches: List[str] = field(default_factory=list)
    total_time: float = 0.0


# =================================================================================
# WIKIBASE CLIENT (INTERNAL)
# =================================================================================

class WikibaseClient:

    def __init__(self, api_url, sparql_url, username, password):
        self.session = requests.Session()
        self.api_url = api_url
        self.sparql_url = sparql_url
        self.username = username
        self.password = password
        self.token = None
        self.login()
        self.csrf = self.get_csrf_token()
    
    def get_csrf_token(self):
        """Get CSRF token for write operations"""
        r = self.session.get(self.api_url, params={
            "action": "query",
            "meta": "tokens",
            "format": "json"
        })
        token = r.json()["query"]["tokens"]["csrftoken"]
        return token

    def login(self):
        """Authenticate with Wikibase"""
        # Step 1: Get login token
        r1 = self.session.get(self.api_url, params={
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json"
        })
        login_token = r1.json()["query"]["tokens"]["logintoken"]

        # Step 2: Login
        r2 = self.session.post(self.api_url, data={
            "action": "login",
            "lgname": self.username,
            "lgpassword": self.password,
            "lgtoken": login_token,
            "format": "json"
        })

        # Step 3: Get CSRF token (edit token)
        r3 = self.session.get(self.api_url, params={
            "action": "query",
            "meta": "tokens",
            "format": "json"
        })
        self.token = r3.json()["query"]["tokens"]["csrftoken"]

    def sparql_query(self, query):
        """Execute SPARQL query"""
        r = requests.get(self.sparql_url,
                         params={"query": query, "format": "json"})
        return r.json()


# =================================================================================
# CONTEXT DETERMINER
# =================================================================================

class ContextDeterminer:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        print("\n" + "="*80)
        print("CONTEXT DETERMINER")
        print("="*80)
        
        # Quick heuristic check BEFORE calling LLM
        #internal_keywords = [
        #     "our", "ours", "kita", "kami",  # possessive
        #     "meeting", "rapat", "contract", "kontrak",  # documents
        #     "project", "hibah", "grant",  # projects
        #     "dosen", "lecturer", "professor",  # people
        #     "telkom university", "telkom",  # institution
        # ] 
        
        # question_lower = question.lower()
        # has_internal_keyword = any(kw in question_lower for kw in internal_keywords)
        
        # if has_internal_keyword:
        # # Still need to check if the question ALSO asks for external info
        # # e.g. "How does our AI research compare to what MIT does globally?"
        #     external_comparison_keywords = ["compare", "global", "worldwide", "industry", "versus", "vs", "bandingkan"]
        #     also_needs_external = any(kw in question_lower for kw in external_comparison_keywords)
            
        #     print(f"✓ Quick route: Detected internal keywords - {'checking external too' if also_needs_external else 'skipping external search'}")
        
        #     if not also_needs_external:
        #         return {
        #             "can_answer_internally": True,
        #             "needs_external_context": False,
        #             "external_search_query": None,
        #             "reasoning": "Question contains internal keywords only"
        #         }
    # else: fall through to LLM for proper hybrid analysis
        
        # Otherwise, use LLM for analysis
        system_prompt = """You are a context analyzer for a multi-database system.

        AVAILABLE DATABASES:
        1. MySQL: Structured data about OUR lecturers, courses, projects, meetings, contracts
        2. FAISS: OUR document content (meeting minutes, contracts, papers)
        3. Wikibase: OUR knowledge graph with lecturers, papers, patents, partnerships
        4. MONGODB: OUR partnership news collection

        WHEN TO USE EXTERNAL WIKIDATA:
        ✓ General knowledge: "Who is Einstein?", "What is AI?", "Tell me about MIT"
        ✓ Well-known entities: famous people, places, organizations, concepts
        ✓ Definitions: "What is quantum computing?", "Explain blockchain"
        ✓ HYBRID: "How does our AI research compare to global trends?" — needs both internal data AND external context

        WHEN NOT TO USE WIKIDATA (use internal DBs only):
        ✗ Questions about OUR data with dates: "meeting in January", "rapat minggu lalu"
        ✗ Questions about OUR entities: "our projects", "hibah wikidata", "Dr. Smith's contract"
        ✗ Questions with possessive pronouns: "our", "kita", "kami" (unless also asking for external comparison)

        ⚠️ IMPORTANT: `can_answer_internally` and `needs_external_context` are INDEPENDENT flags.
        Both can be true at the same time for hybrid questions like:
        - "How does our university's AI research compare to MIT?"
        - "What do our lecturers publish about topics that are globally trending?"
        - "Explain blockchain, and check if we have any related projects"

        RESPOND ONLY WITH JSON:
        {
            "can_answer_internally": true/false,
            "needs_external_context": true/false,
            "external_search_query": "query for Wikidata" or null,
            "reasoning": "brief explanation"
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL_CONTEXT_DETERMINER,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Analyze: "{question}"'}
                ],
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content.strip())
            print(f"\n✓ Can answer with existing DBs: {result['can_answer_internally']}")
            print(f"✓ Needs external context: {result['needs_external_context']}")
            if result['needs_external_context']:
                print(f"✓ External search query: {result['external_search_query']}")
            print(f"✓ Reasoning: {result['reasoning']}")

            return result
        except Exception as e:
            print(f"❌ Context Error: {e}")
            return {"can_answer_internally": True, "needs_external_context": False, "external_search_query": None}


# =================================================================================
# QUERY REASONER
# =================================================================================

class QueryReasoner:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_query_plan(self, question: str, context_info: Dict) -> QueryPlan:
        """Generate a plan for which databases to query and what questions to ask each"""
        print("\n" + "="*80)
        print("QUERY REASONER")
        print("="*80)
        
        system_prompt = f"""You are a query planning expert for a multi-database system.

AVAILABLE DATABASES:

1. MySQL - Structured relational data:
{MYSQL_SCHEMA}

2. FAISS - Vector search on document content:
{FAISS_SCHEMA}
   IMPORTANT: FAISS has TWO query strategies:
   a) "sql_first": Search MySQL for specific entities first (by title, name, ID), 
      then fetch their FAISS documents by faiss_id
      - Use when: Question mentions SPECIFIC document titles, meeting names, or person names
      - Example: "What was in the Budget Review meeting?" → Find meeting in MySQL first
   
   b) "faiss_direct": Direct semantic search in FAISS, then enrich with SQL metadata
      - Use when: Question is about TOPICS, CONCEPTS, or THEMES
      - Example: "What meetings discussed AI?" → Semantic search for AI content
   
   Choose the strategy based on whether the question mentions:
   - SPECIFIC entities (names, titles) → sql_first
   - GENERAL topics/concepts → faiss_direct

3. Wikibase - Knowledge graph:
{WIKIBASE_SCHEMA}

4. MongoDB - Partnership news collection:
{MONGODB_PARTNERSHIP_SCHEMA}

Fields used by this system:
  - partner_name  (str)  : Full name of the partner institution
                           e.g. "Universitas Borneo Tarakan"
  - title         (str)  : Page / article title
                           e.g. "Penandatanganan MOU Telkom University & ..."
  - summary       (str)  : Short auto-generated summary of the article
  - clean_text    (str)  : Full cleaned article body (used for FAISS content)
  - faiss_id      (int)  : ID into the shared FAISS vector index
  - partner_id    (int)  : Numeric ID for the partner institution
**CRITICAL** all partner_name is already affiliated with telkom university, so no need to check for affiliation in MongoDB
for questions asking partnerships with tekkom university, we can directly search for partner_name in MongoDB without needing to check affiliation. 
IMPORTANT: MongoDB has TWO query strategies:
     a) "mongo_first": Filter MongoDB for a specific partner/keyword,
             then fetch FAISS content for those docs.
             Use when: Question names a SPECIFIC partner institution or event.
             Example: "What partnerships does Telkom have with Borneo?"
     b) "faiss_direct": Semantic FAISS search, then enrich with MongoDB metadata.
             Use when: Question is about a TOPIC or THEME in partnership news.
             Example: "What kind of research collaborations has Telkom announced?"
5. Wikidata (EXTERNAL) - Already fetched if needed:
   - General world knowledge, famous entities, definitions, global trends
   - You do NOT need to plan a Wikidata query — it's handled separately
   - But KNOW that if external context is coming, your internal queries 
     should be planned to COMPLEMENT it, not duplicate it
   - Example: If Wikidata will provide general info about "blockchain", 
     your internal queries should focus on OUR blockchain-related projects/docs

YOUR TASK:
Determine which database(s) to query and generate SPECIFIC sub-questions for each.

CRITICAL RULES:
- Each sub-question must be SELF-CONTAINED and answerable by that database alone
- Sub-questions should be MORE DETAILED than the original question, adding context
- Include ALL relevant details from the original question in each sub-question
- Add database-specific context to help generate accurate queries
- DO NOT make sub-questions shorter or vaguer than the original
- If external Wikidata context is also being fetched, plan internal queries 
  to retrieve OUR specific data that complements the external knowledge
- Do NOT try to answer general knowledge questions via internal DBs 
  when Wikidata is already handling that

EXAMPLES:

Main Q: "What papers has Dr. Smith from Telkom University published?"
→ MySQL: "Find the lecturer ID and complete details for Dr. Smith who works at Telkom University, including their name, faculty, and position"
→ Wikibase: "Using the lecturer Dr. Smith from Telkom University, retrieve all research papers (with titles and publication info) that this lecturer has authored or co-authored, linked via the 'Has Researched' property"
(Both needed to connect institutional data with research output)

Main Q: "What was discussed about AI in our meetings?"
→ FAISS: "Search for all meeting document content that discusses or mentions artificial intelligence, AI, machine learning, or related topics, and retrieve the full meeting text"
→ MySQL: "Get the complete meeting metadata (including dates, titles, attendees, project IDs) for meetings that contain discussions about artificial intelligence"
(FAISS for content search, MySQL for metadata)

Main Q: "How many professors teach Machine Learning?"
→ MySQL: "Count the total number of lecturers (dosen) who teach any course (matakuliah) that has 'Machine Learning' or 'ML' in the course name, and include their details"
(Only MySQL needed)

Main Q: "What partnerships does Telkom University have?"
→ Wikibase: "Find all institutional partnerships (using the 'Has Partnership' property P6) for Telkom University, including partner names, types, and any additional relationship details available in the knowledge graph"
(Only Wikibase needed)

Main Q: "siapa dosen yang pernah riset bareng?"
→ Wikibase: "Find all lecturers in the knowledge graph who have co-authored research papers together - identify papers with multiple authors (lecturers linked via 'Has Researched' property P1 to the same paper entity), and return both the lecturer names and the papers they collaborated on"
(Only Wikibase needed - looking for collaboration patterns)

RESPOND ONLY WITH VALID JSON:
{{
    "needs_mysql": true/false,
    "needs_faiss": true/false,
    "needs_wikibase": true/false,
    "mysql_question": "specific question for MySQL" or null,
    "faiss_question": "specific question for FAISS" or null,
    "faiss_strategy": "sql_first" | "faiss_direct" | null,
    "wikibase_question": "specific question for Wikibase" or null,
    "reasoning": "why these databases, questions, and strategies",
    "needs_mongodb": true/false,
    "mongodb_question": "specific question for MongoDB" or null,
    "mongodb_strategy": "mongo_first" | "faiss_direct" | null
}}"""

        try:
            user_content = f'Create a query plan for this question:\n\n"{question}"'
            if context_info.get("needs_external_context"):
                user_content  += (
                    f'\n\nIMPORTANT: This is a HYBRID question. '
                    f'External Wikidata is already being searched for: "{{context_info["external_search_query"]}}". '
                    f'Plan ONLY internal database queries that retrieve OUR specific data '
                    f'to complement the external results.'
                )
            
            response = self.client.chat.completions.create(
                model=MODEL_QUERY_REASONER,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            plan = QueryPlan(
                needs_mysql=result.get("needs_mysql", False),
                needs_faiss=result.get("needs_faiss", False),
                needs_wikibase=result.get("needs_wikibase", False),
                needs_wikidata=context_info.get("needs_external_context", False),
                mysql_question=result.get("mysql_question"),
                faiss_question=result.get("faiss_question"),
                wikibase_question=result.get("wikibase_question"),
                wikidata_query=context_info.get("external_search_query"),
                reasoning=result.get("reasoning", ""),
                needs_mongodb=result.get("needs_mongodb", False),
                mongodb_question=result.get("mongodb_question"),
                mongodb_strategy=result.get("mongodb_strategy", "mongo_first")
            )
            
            # Store FAISS strategy in plan
            plan.faiss_strategy = result.get("faiss_strategy", "faiss_direct")
            plan.needs_mongodb = result.get("needs_mongodb", False)
            plan.mongodb_question = result.get("mongodb_question")
            plan.mongodb_strategy = result.get("mongodb_strategy", "mongo_first")
            
            print(f"\n✓ Query Plan Generated:")
            print(f"  - MySQL: {'YES' if plan.needs_mysql else 'NO'}")
            if plan.needs_mysql:
                print(f"    Question: {plan.mysql_question}")
            print(f"  - FAISS: {'YES' if plan.needs_faiss else 'NO'}")
            if plan.needs_faiss:
                print(f"    Question: {plan.faiss_question}")
                print(f"    Strategy: {plan.faiss_strategy}")
            print(f"  - MongoDB: {'YES' if plan.needs_mongodb else 'NO'}")
            if plan.needs_mongodb:
                print(f"    Question: {plan.mongodb_question}")
                print(f"    Strategy: {plan.mongodb_strategy}")
            print(f"  - Wikibase: {'YES' if plan.needs_wikibase else 'NO'}")
            if plan.needs_wikibase:
                print(f"    Question: {plan.wikibase_question}")
            print(f"  - Wikidata: {'YES' if plan.needs_wikidata else 'NO'}")
            if plan.needs_wikidata:
                print(f"    Query: {plan.wikidata_query}")
            print(f"\n✓ Reasoning: {plan.reasoning}")

            return plan
            
        except Exception as e:
            print(f"❌ Error in query reasoning: {e}")
            return QueryPlan(
                needs_mysql=False,
                needs_faiss=False,
                needs_wikibase=False,
                needs_wikidata=False,
                needs_mongodb=False,
                mysql_question=None,
                faiss_question=None,
                wikibase_question=None,
                wikidata_query=None,
                mongodb_question=None,
                mongodb_strategy=None,
                reasoning=f"Error: {e}"
            )





# =================================================================================
# EXECUTORS (MySQL, FAISS, Internal Wikibase)
# =================================================================================

class MySQLQueryExecutor:
    def __init__(self, client: OpenAI, db_config: Dict):
        self.client = client
        self.db_config = db_config
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        prompt = f"""You are an expert SQL query generator.

Database schema:
{MYSQL_SCHEMA}

Rules:
1. Generate ONLY valid MySQL SQL
2. Do NOT explain anything
3. Do NOT use markdown
4. Use JOINs when needed
5. Only SELECT queries are allowed
6. ALWAYS For name/title search use LIKE with % wildcards for flexibility
7. Always include wikidb. prefix before table names
8. PRESERVE original terms from the question (don't translate "hibah" to "grant", "wikidata" stays "wikidata")
9. Search for the EXACT words mentioned in the question
10. Always LIMIT results to 5 rows, Unless asked otherwise
11.. for WHERE conditions text matching use to lowercase so that "s3" and "S3" are the same
12. for questions asking for counts, return a single row with the count and label the column as count

Examples:
- "meeting hibah wikidata" → WHERE title LIKE '%hibah%' AND title LIKE '%wikidata%'
- "rapat tentang AI" → WHERE title LIKE '%AI%' OR title LIKE '%rapat%'
- "grant meeting" → WHERE title LIKE '%grant%' AND title LIKE '%meeting%'

User question:
{question}

Rules:
1. Generate ONLY valid MySQL SQL
2. Do NOT explain anything
3. Do NOT use markdown
4. Use JOINs when needed
5. Only SELECT queries are allowed
6. ALWAYS For name/title search use LIKE with % wildcards for flexibility
7. Always include wikidb. prefix before table names
8. PRESERVE original terms from the question (don't translate "hibah" to "grant", "wikidata" stays "wikidata")
9. Search for the EXACT words mentioned in the question
10. Always LIMIT results to 5 rows, Unless asked otherwise
11.. for WHERE conditions text matching use to lowercase so that "s3" and "S3" are the same
12. for questions asking for counts, return a single row with the count and label the column as count"""



        response = self.client.chat.completions.create(
            model=MODEL_SQL_GENERATOR,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        # Clean up markdown
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    
    def is_safe_sql(self, sql: str) -> bool:
        """Safety check for SQL queries"""
        forbidden = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE"]
        return not any(keyword in sql.upper() for keyword in forbidden)
    
    def execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL query against MySQL database"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    
    def query(self, question: str, stats: ExecutionStats) -> List[Dict]:
        """Main query method - generate SQL and execute"""
        print("\n" + "-"*80)
        print("MySQL EXECUTOR")
        print("-"*80)
        print(f"Question: {question}")
        
        try:
            # Generate SQL
            sql = self.generate_sql(question)
            print(f"\nGenerated SQL:\n{sql}")
            stats.llm_calls += 1
            stats.mysql_queries.append(sql)
            
            # Safety check
            if not self.is_safe_sql(sql):
                print("❌ Unsafe SQL detected!")
                return []
            
            # Execute
            results = self.execute_sql(sql)
            print(f"✓ Retrieved {len(results)} rows")
            return results
            
        except Exception as e:
            print(f"❌ MySQL execution error: {e}")
            return []

class FAISSQueryExecutor:
    """
    FAISS semantic search executor - refactored from query_faiss.py
    Supports two-phase routing:
    1. SQL-first: Get faiss_ids from MySQL, then fetch documents
    2. FAISS-direct: Semantic search, optionally enrich with SQL metadata
    """
    
    def __init__(self, client: OpenAI, index_path: str, metadata_path: str, db_config: Dict):
        self.client = client
        self.db_config = db_config
        self.index = None
        self.metadata = None
        self.load_index(index_path, metadata_path)
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata"""
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            model=MODEL_EMBEDDING,
            input=text
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def get_documents_by_faiss_ids(self, faiss_ids: List[str]) -> List[Dict]:
        """
        Retrieve FAISS documents by their faiss_id
        Used in SQL-first strategy
        """
        print(f"  Fetching FAISS documents for IDs: {faiss_ids}")
        results = []
        for faiss_id in faiss_ids:
            for doc in self.metadata:
                if str(doc.get('faiss_id')) == str(faiss_id):
                    results.append(doc.copy())
                    
                    # Debug: show document structure and content preview
                    print(f"    ✓ Document {faiss_id} found")
                    print(f"      Fields: {list(doc.keys())}")
                    
                    # Show content preview
                    if 'document_content' in doc:
                        content = doc['document_content']
                        if isinstance(content, str) and len(content) > 0:
                            preview = content[:150].replace('\n', ' ')
                            print(f"      Content preview: {preview}...")
                            print(f"      Content length: {len(content)} characters")
                        else:
                            print(f"      ⚠️  document_content field exists but is empty or not a string")
                    else:
                        print(f"      ⚠️  No 'document_content' field found")
                    break
            else:
                print(f"    ❌ Document {faiss_id} NOT found in metadata")
        
        print(f"  ✓ Successfully retrieved {len(results)}/{len(faiss_ids)} documents")
        return results
    
    def query_direct(self, question: str, stats: ExecutionStats, top_k: int = FAISS_TOP_K) -> List[Dict]:
        """
        Direct semantic search against FAISS
        Strategy: FAISS-direct
        """
        print(f"  Strategy: FAISS-direct semantic search")
        
        try:
            if not self.index or not self.metadata:
                print("❌ FAISS index not loaded")
                return []
            
            # Generate embedding
            query_embedding = self.get_embedding(question)
            stats.llm_calls += 1
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Get results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    doc = self.metadata[idx].copy()
                    doc['similarity_score'] = float(1 / (1 + distances[0][i]))
                    doc['distance'] = float(distances[0][i])
                    doc['rank'] = i + 1
                    results.append(doc)
            
            print(f"  ✓ Retrieved {len(results)} documents via semantic search")
            return results
            
        except Exception as e:
            print(f"❌ FAISS semantic search error: {e}")
            return []
    
    def query_sql_first(self, question: str, stats: ExecutionStats) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        SQL-first strategy: Find specific entities in MySQL, then get their FAISS documents
        Returns: (faiss_documents, sql_metadata)
        """
        print(f"  Strategy: SQL-first (metadata → documents)")
        
        # Analyze what to search for in SQL
        analysis_prompt = f"""Analyze this question to determine SQL search strategy.

Question: "{question}"

Determine:
1. What table to search: meetingminutes, employeecontract, or dosen
2. What field to search in
3. What value to search for

RESPOND ONLY WITH JSON:
{{
    "table": "meetingminutes" | "employeecontract" | "dosen",
    "search_field": "field name",
    "search_value": "value to search",
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_QUERY_REASONER,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content.strip())
            stats.llm_calls += 1
            
            print(f"  SQL Analysis:")
            print(f"    Table: {analysis['table']}")
            print(f"    Field: {analysis['search_field']}")
            print(f"    Value: {analysis['search_value']}")
            
            # Execute SQL query
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            table = analysis['table']
            field = analysis['search_field']
            value = analysis['search_value']
            
            if table == "meetingminutes":
                query = f"SELECT * FROM wikidb.meetingminutes WHERE {field} LIKE %s"
                cursor.execute(query, (f"%{value}%",))
                sql_results = cursor.fetchall()
                faiss_ids = [str(r['faiss_id']) for r in sql_results if r.get('faiss_id')]
                sql_metadata = {"meetings": sql_results, "contracts": [], "dosen": []}
                
            elif table == "employeecontract":
                query = f"""
                    SELECT ec.*, d.nama, d.jabatanAkademik 
                    FROM wikidb.employeecontract ec
                    LEFT JOIN wikidb.dosen d ON ec.IdDosen = d.id
                    WHERE {field} LIKE %s
                """
                cursor.execute(query, (f"%{value}%",))
                sql_results = cursor.fetchall()
                faiss_ids = [str(r['faiss_id']) for r in sql_results if r.get('faiss_id')]
                sql_metadata = {"meetings": [], "contracts": sql_results, "dosen": []}
                
            elif table == "dosen":
                # First get dosen
                query = f"SELECT * FROM wikidb.dosen WHERE {field} LIKE %s"
                cursor.execute(query, (f"%{value}%",))
                dosen_results = cursor.fetchall()
                
                # Then get their contracts
                if dosen_results:
                    dosen_ids = [d['id'] for d in dosen_results]
                    placeholders = ','.join(['%s'] * len(dosen_ids))
                    query = f"SELECT * FROM wikidb.employeecontract WHERE IdDosen IN ({placeholders})"
                    cursor.execute(query, dosen_ids)
                    contract_results = cursor.fetchall()
                    faiss_ids = [str(c['faiss_id']) for c in contract_results if c.get('faiss_id')]
                    sql_metadata = {"meetings": [], "contracts": contract_results, "dosen": dosen_results}
                else:
                    faiss_ids = []
                    sql_metadata = {"meetings": [], "contracts": [], "dosen": []}
            
            cursor.close()
            conn.close()
            
            print(f"  ✓ Found {len(faiss_ids)} faiss_ids from SQL")
            stats.mysql_queries.append(query)
            
            # Get FAISS documents by IDs
            faiss_documents = self.get_documents_by_faiss_ids(faiss_ids) if faiss_ids else []
            
            return faiss_documents, sql_metadata
            
        except Exception as e:
            print(f"❌ SQL-first query error: {e}")
            return [], {"meetings": [], "contracts": [], "dosen": []}
    
    def enrich_with_metadata(self, faiss_results: List[Dict], stats: ExecutionStats) -> Dict[str, List[Dict]]:
        """
        Enrich FAISS results with SQL metadata
        Used after FAISS-direct search
        """
        print(f"  Enriching {len(faiss_results)} documents with SQL metadata...")
        
        if not faiss_results:
            return {"meetings": [], "contracts": [], "dosen": []}
        
        faiss_ids = [str(doc.get('faiss_id')) for doc in faiss_results if doc.get('faiss_id')]
        
        if not faiss_ids:
            return {"meetings": [], "contracts": [], "dosen": []}
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            placeholders = ','.join(['%s'] * len(faiss_ids))
            sql_metadata = {"meetings": [], "contracts": [], "dosen": []}
            
            # Get meetings
            query = f"SELECT * FROM wikidb.meetingminutes WHERE faiss_id IN ({placeholders})"
            cursor.execute(query, faiss_ids)
            sql_metadata["meetings"] = cursor.fetchall()
            
            # Get contracts with dosen info
            query = f"""
                SELECT ec.*, d.nama, d.jabatanAkademik, d.pendidikanTertinggi
                FROM wikidb.employeecontract ec
                LEFT JOIN wikidb.dosen d ON ec.IdDosen = d.id
                WHERE ec.faiss_id IN ({placeholders})
            """
            cursor.execute(query, faiss_ids)
            sql_metadata["contracts"] = cursor.fetchall()
            
            # Get dosen info
            if sql_metadata["contracts"]:
                dosen_ids = list(set([c['IdDosen'] for c in sql_metadata["contracts"] if c.get('IdDosen')]))
                if dosen_ids:
                    dosen_placeholders = ','.join(['%s'] * len(dosen_ids))
                    query = f"SELECT * FROM wikidb.dosen WHERE id IN ({dosen_placeholders})"
                    cursor.execute(query, dosen_ids)
                    sql_metadata["dosen"] = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            print(f"  ✓ Enriched with {len(sql_metadata['meetings'])} meetings, "
                  f"{len(sql_metadata['contracts'])} contracts, {len(sql_metadata['dosen'])} dosen")
            
            return sql_metadata
            
        except Exception as e:
            print(f"❌ Metadata enrichment error: {e}")
            return {"meetings": [], "contracts": [], "dosen": []}
    
    def query(self, question: str, strategy: str, stats: ExecutionStats, mysql_results: List[Dict] = None) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Main query method with routing strategy
        
        Args:
            question: Natural language question
            strategy: "sql_first" or "faiss_direct"
            stats: Execution statistics tracker
            mysql_results: Optional pre-fetched MySQL results to extract faiss_ids from
            
        Returns:
            (faiss_documents, sql_metadata)
        """
        print("\n" + "-"*80)
        print("FAISS EXECUTOR")
        print("-"*80)
        print(f"Question: {question}")
        
        stats.faiss_queries.append(question)
        
        if strategy == "sql_first":
            # Check if we already have MySQL results with faiss_ids
            if mysql_results:
                print(f"  Strategy: Using pre-fetched MySQL results")
                
                # Extract faiss_ids from MySQL results
                faiss_ids = []
                for result in mysql_results:
                    if result.get('faiss_id'):
                        faiss_ids.append(str(result['faiss_id']))
                
                if faiss_ids:
                    print(f"  ✓ Extracted {len(faiss_ids)} faiss_ids from MySQL results: {faiss_ids}")
                    
                    # Get FAISS documents by IDs
                    faiss_documents = self.get_documents_by_faiss_ids(faiss_ids)
                    
                    # SQL metadata is already in mysql_results
                    sql_metadata = {"meetings": [], "contracts": [], "dosen": []}
                    
                    # Categorize MySQL results
                    for result in mysql_results:
                        # Check which table this result is from based on fields
                        if 'ProjectID' in result or ('title' in result and 'date' in result):  # meetingminutes
                            sql_metadata["meetings"].append(result)
                        elif 'baseSalary' in result:  # employeecontract
                            sql_metadata["contracts"].append(result)
                        elif 'jabatanAkademik' in result:  # dosen
                            sql_metadata["dosen"].append(result)
                    
                    return faiss_documents, sql_metadata
                else:
                    print(f"  ⚠️  MySQL results exist but contain no faiss_ids")
                    print(f"  ⚠️  Cannot fetch FAISS documents without faiss_ids")
                    return [], {"meetings": mysql_results, "contracts": [], "dosen": []}
            else:
                print(f"  ⚠️  No pre-fetched MySQL results available")
                print(f"  ⚠️  SQL-first strategy requires MySQL results first")
                print(f"  ⚠️  Skipping FAISS retrieval - no documents to fetch")
                return [], {"meetings": [], "contracts": [], "dosen": []}
            
        elif strategy == "faiss_direct":
            # FAISS-direct: Semantic search, then enrich with metadata
            faiss_results = self.query_direct(question, stats)
            sql_metadata = self.enrich_with_metadata(faiss_results, stats)
            return faiss_results, sql_metadata
            
        else:
            print(f"❌ Unknown FAISS strategy: {strategy}")
            return [], {"meetings": [], "contracts": [], "dosen": []}

class InternalWikibaseExecutor:
    """
    Wikibase SPARQL query executor - refactored from query_wikibase.py
    Uses WikibaseClient for authentication and queries
    """
    
    def __init__(self, client: OpenAI, wikibase_client: WikibaseClient):
        self.client = client
        self.wikibase_client = wikibase_client
    
    def generate_sparql(self, question: str) -> str:
        """Generate SPARQL query from natural language question"""
        prompt = f"""
{WIKIBASE_SCHEMA}

You are a SPARQL Query Generator for a Wikibase Cloud instance. Your goal is to generate precise, working SPARQL queries that retrieve Lecturers, Papers, and Partnerships.

────────────────────────────────────────
PREFIXES
────────────────────────────────────────
PREFIX wd:       <http://38.147.122.59/entity/>
PREFIX wdt:      <http://38.147.122.59/prop/direct/>
PREFIX p:        <http://38.147.122.59/prop/>
PREFIX ps:       <http://38.147.122.59/prop/statement/>
PREFIX bd:       <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>

────────────────────────────────────────
DATA SCHEMA
────────────────────────────────────────
- P1 (Has Researched): Lecturer -> wdt:P1 -> Paper
- P2 (Has Patent):     Lecturer -> wdt:P2 -> Patent
- P3 (is Lecturer):    Entity   -> wdt:P3 -> [] (Type check)
- P4 (is Paper):       Entity   -> wdt:P4 -> [] (Type check)
- P5 (is Patent):      Entity   -> wdt:P5 -> [] (Type check)
- P6 (Has Partnership): Institution -> wdt:P6 -> Partner
- P7 (Has Faculty):    Institution -> wdt:P7 -> Faculty
- P8 (Affiliation):    Person      -> wdt:P8 -> Organization
- P9 (Has Specialty):  Lecturer    -> wdt:P9 -> Specialty

────────────────────────────────────────
***CRITICAL STRATEGY FOR 0-RESULT PREVENTION****
────────────────────────────────────────
To ensure you find results, you must use this specific search pattern:

1. **SEARCH PHASE (Finding the Subject):**
   - NEVER assume you know the Q-ID (e.g., do not guess wd:Q123).
   - ALWAYS search for the subject by matching its label.
   - Use `FILTER(CONTAINS(LCASE(?label), "search term"))` which is more robust than REGEX.
   - **Important:** Search specifically on `rdfs:label`, NOT `wikibase:label`.

2. **RETRIEVAL PHASE (Traversing):**
   - Once the variable `?subject` is bound, follow the `wdt:P...` paths.

3. **DISPLAY PHASE:**
   - Use `SERVICE wikibase:label` only to get the nice names of the *found* items (the results).

────────────────────────────────────────
STRICT NAME MATCHING RULES FOR LLM SPARQL GENERATION
────────────────────────────────────────

1. Always fetch labels via SERVICE wikibase:label.  
   - Never bind labels directly using rdfs:label outside SERVICE.  

2. Name matching must operate only on variables retrieved from SERVICE.  
   - Example: ?itemLabel from SERVICE.

3. Honorifics must be ignored when matching names.  
   - Common honorifics: "mr.", "mrs.", "ms.", "dr.", "prof.", "pak ", "bu ", "bapak ", "ibu "  
   - Strip honorifics at the start of the label before applying FILTER or REGEX.  
   - Use SPARQL REPLACE or equivalent function to remove these prefixes.

4. Matching must be case-insensitive and flexible:  
   - Use REGEX or CONTAINS with LCASE, applied **after stripping honorifics**.  
   - Example:
     ```sparql
     FILTER(REGEX(REPLACE(LCASE(?itemLabel), "^(mr\\.|mrs\\.|ms\\.|dr\\.|prof\\.|pak\\s|bu\\s|bapak\\s|ibu\\s)", ""), "kemas", "i"))
     ```

5. Do NOT include honorifics in the query string.  
   - If the user asks for "Pak Kemas", the LLM should generate a query for `"kemas"` only.  

6. Always use minimal triple patterns required to answer the question.  
   - Do not add unnecessary joins, UNIONs, or rdfs:label bindings outside SERVICE.  

7. Respect case-insensitive filtering across all name-based queries.

8. This rule applies to **all name searches** in the KG, whether subjects, objects, or people labels.  


────────────────────────────────────────
QUERY RULES
────────────────────────────────────────
1. **Partnerships (P6):** - When looking for "partners of Telkom University", first find "Telkom University" via label match.
   - Then use `?uni wdt:P6 ?partner` to get the list.
   
2. **Limits:**
   - If the user asks for a number (e.g., "3 partners"), add `LIMIT 3` at the end of the query.

3. **Output Format:**
   - Return ONLY the SPARQL code block. No text, no markdown, no explanations.

4. ALWAYS LIMIT TO 5, UNLESS SPECIFIED OTHERWISE.


────────────────────────────────────────
CURRENT QUESTION
────────────────────────────────────────
{question}
```

Question: {question}"""

        response = self.client.chat.completions.create(
            model=MODEL_SPARQL_GENERATOR,
            messages=[
                {"role": "system", "content": "You are a SPARQL query expert. Generate only valid SPARQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        sparql = response.choices[0].message.content.strip()
        sparql = sparql.replace("```sparql", "").replace("```", "").strip()
        return sparql
    
    def query(self, question: str, stats: ExecutionStats) -> List[Dict]:
        """Execute SPARQL query against Wikibase"""
        print("\n" + "-"*80)
        print("WIKIBASE EXECUTOR")
        print("-"*80)
        print(f"Question: {question}")
        
        try:
            # Generate SPARQL
            sparql = self.generate_sparql(question)
            print(f"\nGenerated SPARQL:\n{sparql}")
            stats.llm_calls += 1
            stats.wikibase_queries.append(sparql)
            
            # Execute using WikibaseClient
            results = self.wikibase_client.sparql_query(sparql)
            bindings = results.get('results', {}).get('bindings', [])
            
            print(f"✓ Retrieved {len(bindings)} results")
            return bindings
            
        except Exception as e:
            print(f"❌ Wikibase execution error: {e}")
            return []



# =================================================================================
# WIKIDATA SEARCHER (EXTERNAL - LLM SPARQL)
# =================================================================================

import requests
from typing import List, Dict


class WikidataSearcher:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.serpapi_endpoint = "https://serpapi.com/search"
        self.headers = {
            "User-Agent": "UnifiedRAGSystem/2.0 (internal-research; contact@example.com)",
            "Accept": "application/json"
        }
        self.timeout = 30

    def search(self, question: str, stats) -> str:
        """
        Dual search:
          1. question + 'Wikidata' → top 3 results
          2. question alone        → top 2 results
        Results are deduplicated by URL and merged.
        """
        print("\n" + "-" * 80)
        print("WIKIDATA SEARCH  (via SerpAPI dual-pass)")
        print("-" * 80)
        print(f"Question: {question}")

        # ── Pass 1: with Wikidata appended ─────────────────────────────
        query_wiki = f"{question} Wikidata"
        print(f"\n[Pass 1] Searching: '{query_wiki}'")
        results_wiki = self._serpapi_search(query_wiki, num=3)
        stats.wikidata_searches.append(f"serpapi_pass1: {query_wiki}")

        # ── Pass 2: plain question ──────────────────────────────────────
        print(f"[Pass 2] Searching: '{question}'")
        results_plain = self._serpapi_search(question, num=2)
        stats.wikidata_searches.append(f"serpapi_pass2: {question}")

        # ── Merge & deduplicate by URL ──────────────────────────────────
        seen_urls = set()
        merged = []
        for r in results_wiki + results_plain:
            url = r.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(r)

        print(f"  ✓ Merged {len(merged)} unique results "
              f"({len(results_wiki)} wikidata + {len(results_plain)} plain, deduped)")

        if not merged:
            return "No information found for this query."

        return self._format_snippets(merged)

    # ------------------------------------------------------------------
    # SerpAPI web search
    # ------------------------------------------------------------------

    def _serpapi_search(self, query: str, num: int = 3) -> List[Dict]:
        try:
            response = requests.get(
                self.serpapi_endpoint,
                params={
                    "q":       query,
                    "api_key": self.serpapi_key,
                    "engine":  "google",
                    "num":     num,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json().get("organic_results", [])[:num]
            print(f"  ✓ Returned {len(results)} results")
            return results
        except requests.exceptions.Timeout:
            print(f"❌ SerpAPI timeout for query: {query}")
            return []
        except requests.exceptions.HTTPError as e:
            print(f"❌ SerpAPI HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:
            print(f"❌ SerpAPI unexpected error: {type(e).__name__}: {e}")
            return []

    # ------------------------------------------------------------------
    # Format snippets
    # ------------------------------------------------------------------

    def _format_snippets(self, results: List[Dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            link    = r.get("link", "").strip()
            if title and snippet:
                lines.append(
                    f"[Result {i}] {title}\n"
                    f"  {snippet}\n"
                    f"  Source: {link}"
                )
        result = "\n\n".join(lines)
        print(f"  ✓ Formatted {len(lines)} snippets")
        return result if result else "No information found."
# =================================================================================
# MONGODB PARTNERSHIP EXECUTOR
# =================================================================================       
class MongoDBPartnershipExecutor:
    """
    Retrieves partnership news from MongoDB and optionally cross-references
    the shared FAISS index.

    Supports two strategies (mirroring FAISSQueryExecutor):
      - "mongo_first"   : Query MongoDB for specific partners / keywords,
                          then fetch their FAISS documents by faiss_id.
      - "faiss_direct"  : Semantic FAISS search first, then enrich with
                          MongoDB metadata for partnership documents only.

    Constructor parameters
    ----------------------
    client       : OpenAI client (shared with the rest of the system)
    faiss_executor : the FAISSQueryExecutor instance already created in
                     UnifiedRAGSystem — we reuse its index + get_embedding()
    mongo_config : dict with keys: host, port, username, password,
                   database, collection
    model        : LLM model name used for query analysis
    """

    def __init__(
        self,
        client: OpenAI,
        faiss_executor,                 # FAISSQueryExecutor — passed in, not imported
        mongo_config: Dict = None,
        model: str = "gpt-4.1-mini",
    ):
        self.client = client
        self.faiss_executor = faiss_executor
        self.model = model
        self.mongo_config = mongo_config or MONGO_CONFIG
        self._mongo_client: Optional[MongoClient] = None
        self._collection = None
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Establish MongoDB connection (lazy, call once at init)."""
        try:
            cfg = self.mongo_config
            if cfg.get("username") and cfg.get("password"):
                from urllib.parse import quote_plus
                username = quote_plus(cfg["username"])
                password = quote_plus(cfg["password"])   # handles @ % and other special chars
                uri = (
                    f"mongodb://{username}:{password}"
                    f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
                    f"?authSource=admin"
                )
            else:
                uri = f"mongodb://{cfg['host']}:{cfg['port']}"

            self._mongo_client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )
            # Ping BEFORE assigning _collection so failure stays clean
            self._mongo_client.admin.command("ping")

            db = self._mongo_client[cfg["database"]]
            self._collection = db[cfg["collection"]]

            count = self._collection.count_documents({})
            print(f"✓ MongoDB connected: {cfg['database']}.{cfg['collection']} ({count} documents)")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self._collection = None

    def _project(self) -> Dict:
        """MongoDB projection — only return the fields we care about."""
        return {
            "_id": 0,
            "partner_name": 1,
            "title": 1,
            "summary": 1,
            "clean_text": 1,
            "faiss_id": 1,
            "partner_id": 1,
        }

    def _serialize(self, docs: List[Dict]) -> List[Dict]:
        """Ensure all values are JSON-serialisable (ObjectId, datetime, etc.)."""
        import datetime
        safe = []
        for d in docs:
            row = {}
            for k, v in d.items():
                if isinstance(v, datetime.datetime):
                    row[k] = v.isoformat()
                else:
                    row[k] = v
            safe.append(row)
        return safe

    # ------------------------------------------------------------------
    # LLM-assisted query analysis
    # ------------------------------------------------------------------

    def _analyze_question(self, question: str) -> Dict:
        """
        Ask the LLM how to translate a natural-language question into a
        MongoDB filter against the partnership collection.

        Returns a dict:
          {
            "search_field": "partner_name" | "title" | "summary" | "partner_id",
            "search_value": "<value>",
            "use_regex": true | false,
            "reasoning": "..."
          }
        """
        prompt = f"""YYou are a query analysis expert for a MongoDB collection that stores
Telkom University partnership news articles.

{MONGODB_PARTNERSHIP_SCHEMA}

Analyze the question below and decide how to query the collection.

Question: "{question}"

Rules:
- If the question mentions a specific institution / partner name → search partner_name
- If the question is about a topic, keyword, or event type → search title or summary
- If the question mentions a numeric partner ID → search partner_id (exact, no regex)
- For text fields always set use_regex: true so LIKE-style matching works
- If the question is broad (e.g. "show all", "list partnerships", "any partner") →
  set search_field to "title", search_value to "Telkom", use_regex to true

NEVER use ".*" or "*" as a search_value — use a real keyword instead.

RESPOND ONLY WITH VALID JSON (no markdown, no explanation):
{{
    "search_field": "partner_name" | "title" | "summary" | "partner_id",
    "search_value": "<value to search for>",
    "use_regex": true | false,
    "reasoning": "<one-line explanation>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            return {"llm_calls": 1, **__import__("json").loads(raw)}
        except Exception as e:
            print(f"❌ MongoDB query analysis error: {e}")
            # Safe fallback: broad title search
            return {
                "llm_calls": 0,
                "search_field": "title",
                "search_value": question[:60],
                "use_regex": True,
                "reasoning": f"Fallback due to error: {e}",
            }

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def get_documents_by_faiss_ids(self, faiss_ids: List) -> List[Dict]:
        """
        Fetch MongoDB partnership documents whose faiss_id is in faiss_ids.
        Used when FAISS has already found relevant vectors and we want the
        richer MongoDB metadata alongside.
        """
        if self._collection is None or not faiss_ids:
            return []

        # Normalise: MongoDB stores faiss_id as int; FAISS metadata may give strings
        int_ids = []
        for fid in faiss_ids:
            try:
                int_ids.append(int(fid))
            except (ValueError, TypeError):
                pass

        if not int_ids:
            return []

        try:
            cursor = self._collection.find(
                {"faiss_id": {"$in": int_ids}},
                self._project(),
            )
            docs = self._serialize(list(cursor))
            print(f"  ✓ MongoDB: found {len(docs)}/{len(int_ids)} partnership docs by faiss_id")
            return docs
        except Exception as e:
            print(f"❌ MongoDB faiss_id lookup error: {e}")
            return []

    def query_mongo_first(self, question: str, stats) -> Tuple[List[Dict], List[Dict]]:
        """
        Mongo-first strategy:
          1. Use LLM to build a MongoDB filter from the question.
          2. Retrieve matching MongoDB documents (partner_name, title, etc.).
          3. Extract their faiss_ids and pull the full FAISS document content.

        Returns: (faiss_documents, mongo_metadata)
        """
        print(f"  Strategy: mongo_first (MongoDB filter → FAISS documents)")

        if self._collection is None:
            print("❌ MongoDB not connected")
            return [], []

        analysis = self._analyze_question(question)
        stats.llm_calls += analysis.get("llm_calls", 0)

        field = analysis["search_field"]
        value = analysis["search_value"]
        use_regex = analysis.get("use_regex", True)

        print(f"  MongoDB Analysis:")
        print(f"    Field:     {field}")
        print(f"    Value:     {value}")
        print(f"    Regex:     {use_regex}")
        print(f"    Reasoning: {analysis['reasoning']}")

        try:
            if use_regex and field != "partner_id":
                mongo_filter = {field: {"$regex": value, "$options": "i"}}
            else:
                # Exact match (e.g. partner_id integer)
                try:
                    mongo_filter = {field: int(value)}
                except ValueError:
                    mongo_filter = {field: value}

            cursor = self._collection.find(mongo_filter, self._project()).limit(10)
            mongo_docs = self._serialize(list(cursor))

            print(f"  ✓ MongoDB returned {len(mongo_docs)} partnership records")
            

            return [], mongo_docs

        except Exception as e:
            print(f"❌ mongo_first query error: {e}")
            return [], []

    def query_faiss_direct(self, question: str, stats) -> Tuple[List[Dict], List[Dict]]:
        """
        FAISS-direct strategy:
          1. Perform semantic FAISS search (reuses FAISSQueryExecutor.query_direct).
          2. Filter results to only those that exist in the MongoDB partnerships
             collection (i.e. skip meeting-minutes / contract vectors).
          3. Enrich with MongoDB metadata.

        Returns: (faiss_documents_filtered, mongo_metadata)
        """
        print(f"  Strategy: faiss_direct → enrich with MongoDB partnership metadata")

        if not self.faiss_executor or not self.faiss_executor.index:
            print("❌ FAISS executor not available")
            return [], []

        # Run semantic search
        faiss_results = self.faiss_executor.query_direct(question, stats)

        if not faiss_results:
            return [], []

        # Keep only vectors that have a corresponding MongoDB partnership doc
        faiss_ids = [str(d["faiss_id"]) for d in faiss_results if d.get("faiss_id") is not None]
        mongo_docs = self.get_documents_by_faiss_ids(faiss_ids)

        # Build a set of faiss_ids that are actually in MongoDB
        mongo_faiss_ids = {str(d["faiss_id"]) for d in mongo_docs if d.get("faiss_id") is not None}

        # Filter FAISS results to partnership docs only
        filtered_faiss = [d for d in faiss_results if str(d.get("faiss_id")) in mongo_faiss_ids]

        print(
            f"  ✓ Filtered to {len(filtered_faiss)} FAISS docs that are partnership articles "
            f"(from {len(faiss_results)} total semantic results)"
        )

        return filtered_faiss, mongo_docs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        strategy: str,
        stats,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Main query method — mirrors FAISSQueryExecutor.query() in structure.

        Parameters
        ----------
        question  : natural-language question about partnerships
        strategy  : "mongo_first" or "faiss_direct"
        stats     : ExecutionStats instance (from superquery.py)

        Returns
        -------
        (faiss_documents, mongo_documents)
          faiss_documents — raw FAISS doc dicts (with document_content, etc.)
          mongo_documents — MongoDB partnership dicts
                            (partner_name, title, summary, clean_text, …)
        """
        print("\n" + "-" * 80)
        print("MONGODB PARTNERSHIP EXECUTOR")
        print("-" * 80)
        print(f"Question : {question}")
        print(f"Strategy : {strategy}")

        if strategy == "mongo_first":
            return self.query_mongo_first(question, stats)

        elif strategy == "faiss_direct":
            return self.query_faiss_direct(question, stats)

        else:
            print(f"❌ Unknown strategy: {strategy}. Defaulting to mongo_first.")
            return self.query_mongo_first(question, stats)

# =================================================================================
# ANSWER SYNTHESIZER
# =================================================================================

class AnswerSynthesizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def _summarize_chunk(self, question: str, source_label: str, content: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_ANSWER_SYNTHESIZER,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise information extractor. "
                            "Given a source and a question, extract ALL facts "
                            "from the source that are relevant to answering the question. "
                            "IMPORTANT RULES:\n"
                            "- For Partnership News sources: ALWAYS extract partner_name, "
                            "title, and summary — these are always relevant.\n"
                            "- For list/enumeration questions ('show all', 'list', 'what partners'): "
                            "extract ALL items, do not filter or truncate them.\n"
                            "- Preserve all specific values: names, numbers, dates, IDs.\n"
                            "- Only say 'No relevant information found' if the source is "
                            "genuinely empty or completely unrelated to the topic.\n"
                            "- Be thorough. An incomplete extraction is worse than a long one."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Source: {source_label}\n\n"
                            f"Content:\n{content}\n\n"
                            f"Question: {question}\n\n"
                            f"Extract ALL relevant facts (for Partnership News, always include "
                            f"partner_name, title, and summary):"
                        )
                    }
                ],
                max_tokens=1500  # was 500 — gives room for complete extraction
            )
            return f"[{source_label}]\n{resp.choices[0].message.content.strip()}"
        except Exception as e:
            print(f"❌ Chunk summarization error ({source_label}): {e}")
            return f"[{source_label}]\n{content[:2000]}"  # was 500 — preserve more on failure

    def _build_chunks(
        self,
        mysql: List,
        faiss: List,
        wikibase: List,
        external: str,
        mongodb_meta: List
    ) -> List[Tuple[str, str]]:
        chunks = []

        if external:
            chunks.append(("External Knowledge (Wikidata)", external))

        if mysql:
            content = (
                f"The following is the result retrieved from the internal MySQL database "
                f"in response to the question. The data provided already has been filtered "
                f"to answer the question, meaning the answer to the question is the data provided here.\n\n"
                f"{json.dumps(mysql, indent=2, default=str)}"
            )
            chunks.append(("Internal Database (MySQL)", content))

        if wikibase:
            chunks.append(("Internal Knowledge Graph (Wikibase)", json.dumps(wikibase, default=str)))

        if faiss:
            for i, doc in enumerate(faiss):
                chunks.append((f"Internal Document #{i+1}", json.dumps(doc, default=str)))

        if mongodb_meta:
            for i, doc in enumerate(mongodb_meta):
                content = (
                    f"Partner: {doc.get('partner_name', 'Unknown')}\n"
                    f"Title: {doc.get('title', 'Unknown')}\n"
                    f"Summary: {doc.get('summary', '')}\n"
                    f"Full Text: {doc.get('clean_text', '')}"
                )
                chunks.append((f"Partnership News #{i+1}", content))

        return chunks

    def _needs_compression(self, chunks: List[Tuple[str, str]], threshold: int = 20000) -> bool:
        # Raised from 6000 → 20000. Modern models handle large contexts well;
        # compress only when truly necessary to avoid losing information.
        total = sum(len(label) + len(content) for label, content in chunks)
        print(f"  📏 Total context size: {total} chars (threshold: {threshold})")
        return total > threshold

    def synthesize(
        self,
        question: str,
        mysql: List,
        faiss: List,
        wikibase: List,
        external: str,
        mongodb_meta: List,
        stats: ExecutionStats
    ) -> str:
        print("\n" + "=" * 80)
        print("ANSWER SYNTHESIZER")
        print("=" * 80)

        chunks = self._build_chunks(mysql, faiss, wikibase, external, mongodb_meta)

        if not chunks:
            return "I couldn't find any relevant information."

        SKIP_COMPRESSION = {
            "External Knowledge (Wikidata)",
            "Internal Database (MySQL)",
            "Internal Knowledge Graph (Wikibase)"
        }

        if self._needs_compression(chunks):
            print(f"  ⚡ Large context detected ({len(chunks)} chunks) — compressing...")
            summarized = []
            for source_label, content in chunks:
                if source_label in SKIP_COMPRESSION:
                    summarized.append(f"[{source_label}]\n{content}")
                else:
                    summary = self._summarize_chunk(question, source_label, content)
                    summarized.append(summary)
                    stats.llm_calls += 1
            joined_context = "\n\n".join(summarized)
            print(f"  ✓ Compressed to {len(joined_context)} chars")
        else:
            # No compression needed — pass full content directly
            joined_context = "\n\n".join(
                f"[{label}]\n{content}" for label, content in chunks
            )

        print(f"\n📨 FINAL CONTEXT TO LLM ({len(joined_context)} chars, {len(chunks)} sources):")
        print("-" * 80)
        print(joined_context[:3000])
        if len(joined_context) > 3000:
            print(f"\n... [truncated — {len(joined_context) - 3000} more chars not shown]")
        print("-" * 80)

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_ANSWER_SYNTHESIZER,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer based ONLY on the provided context. "
                            "Do NOT use your training knowledge. "
                            "Synthesize a clear, complete answer and cite sources explicitly "
                            "(e.g. 'According to Partnership News #3...' or 'Wikidata reports...'). "
                            "all names given are the names of Lektors,, regardless of their positions"
                            "If the question asks for a list, include ALL items found in the context — "
                            "do not summarize or truncate the list."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{joined_context}\n\n"
                            f"Question: {question}\n\n"
                            f"REMINDER: Answer ONLY from the context above. "
                            f"If the question asks for multiple items, list ALL of them. "
                            f"Do not omit any item found in the context."
                        )
                    }
                ],
                max_tokens=2000  # explicit ceiling so long lists aren't cut off
            )
            stats.llm_calls += 1
            return resp.choices[0].message.content.strip()

        except Exception as e:
            return f"Error synthesizing: {e}"

# =================================================================================
# MAIN SYSTEM
# =================================================================================

class UnifiedRAGSystem:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.wb_client = WikibaseClient(
            api_url=WIKIBASE_CONFIG["api_url"],
            sparql_url=WIKIBASE_CONFIG["sparql_url"],
            username=WIKIBASE_CONFIG["username"],
            password=WIKIBASE_CONFIG["password"]
        )

        self.context_determiner = ContextDeterminer(self.client)
        self.query_reasoner = QueryReasoner(self.client)

        # Executors
        self.mysql = MySQLQueryExecutor(self.client, DB_CONFIG)
        self.faiss = FAISSQueryExecutor(self.client, FAISS_INDEX_PATH, METADATA_PATH, DB_CONFIG)
        self.wikibase = InternalWikibaseExecutor(self.client, self.wb_client)
        self.mongodb = MongoDBPartnershipExecutor(
            client=self.client,
            faiss_executor=self.faiss,
            mongo_config=MONGO_CONFIG,
        )
        # External SPARQL Searcher
        self.wikidata = WikidataSearcher(serpapi_key=SERPAPI_KEY)

        self.synthesizer = AnswerSynthesizer(self.client)

    def ask(self, question: str) -> Tuple[str, ExecutionStats]:
        start = datetime.now()
        stats = ExecutionStats()

        print(f"\n❓ Processing: {question}")

        # 1. Determine Context
        ctx = self.context_determiner.analyze_query(question)
        stats.llm_calls += 1

        mysql_res, faiss_res, wikibase_res = [], [], []
        mongodb_faiss_res, mongodb_meta_res = [], []
        ext_res = ""

        # 2. Route External Queries (Wikidata SPARQL)
        # ✅ Always run if needed — no early return, internal may also be needed
        if ctx["needs_external_context"]:
            search_q = ctx.get("external_search_query")
            if not search_q or len(search_q) < 5:
                search_q = question

            print(f"🌍 Routing to External Wikidata (SPARQL) for: {search_q}")
            ext_res = self.wikidata.search(search_q, stats)

            # ✅ FIX 1: Track external usage so Sources UI shows "External Wikidata"
            if ext_res:
                stats.databases_used.append("External Wikidata")

        # 3. Route Internal Queries
        # ✅ Runs independently — even if external was also fetched
        if ctx["can_answer_internally"]:
            plan = self.query_reasoner.generate_query_plan(question, ctx)
            stats.llm_calls += 1

            if plan.needs_mysql and plan.mysql_question:
                stats.databases_used.append("MySQL")
                mysql_res = self.mysql.query(plan.mysql_question, stats)

            if plan.needs_faiss and plan.faiss_question:
                stats.databases_used.append("FAISS")
                faiss_res, _ = self.faiss.query(
                    plan.faiss_question, plan.faiss_strategy, stats, mysql_res
                )

            if plan.needs_wikibase and plan.wikibase_question:
                stats.databases_used.append("InternalWikibase")
                wikibase_res = self.wikibase.query(plan.wikibase_question, stats)

            if plan.needs_mongodb and plan.mongodb_question:
                stats.databases_used.append("MongoDB")
                mongodb_faiss_res, mongodb_meta_res = self.mongodb.query(
                    plan.mongodb_question,
                    plan.mongodb_strategy or "mongo_first",
                    stats,
                )
                faiss_res.extend(mongodb_faiss_res)

        # ✅ FIX 2: Correct fallback — plain else (original elif condition was logically redundant)
        else:
            print("⚠️  No routing path matched — falling back to synthesizer with empty context")

        # 4. Synthesize Final Answer
        # ext_res is passed regardless — synthesizer handles empty string gracefully
        answer = self.synthesizer.synthesize(
            question, mysql_res, faiss_res, wikibase_res, ext_res, mongodb_meta_res, stats
        )

        stats.total_time = (datetime.now() - start).total_seconds()
        return answer, stats


def main():
    system = UnifiedRAGSystem()
    print("\nSystem Ready. Enter your question (or 'quit').")
    
    while True:
        try:
            q = input("\n>> ").strip()
            if q.lower() in ['quit', 'exit']: break
            if not q: continue
            
            ans, stats = system.ask(q)
            print(f"\n📝 ANSWER:\n{ans}\n")
            print(f"⏱️  {stats.total_time:.2f}s | LLM Calls: {stats.llm_calls} | DBs: {stats.databases_used}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()