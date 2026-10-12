import os
import re
import json
import logging
import mimetypes
from html.parser import HTMLParser
from http.client import HTTPException
from typing import TypedDict, Annotated, List
from urllib.error import URLError
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
 
mimetypes.add_type("image/avif", ".avif")

# --- only the web layer is new (FastAPI replaces Streamlit) ---
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
 
load_dotenv()
 
# ----------------------------
# Step 1 - Building the RAG retrievers (runs once at server start)
# ----------------------------
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
    def build_retriver(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
 
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 8})
 
    acedemic_retriever = build_retriver("academics_handbook.pdf")
    fee_retriever = build_retriver("fee_structure.pdf")
 
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
 
    return acedemic_retriever, fee_retriever, llm
 
 
acedemic_retriever, fee_retriever, llm = load_resources()
 
 
# ----------------------------
# Step 2 - State
# ----------------------------
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str
    images_shown: str
 
 
# ----------------------------
# Image library (photos taken from the SR University PDFs, stored in ./images)
# (id, file, caption, tags) - order = priority
# ----------------------------
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
IMAGE_CATALOG = [
    ["hostel-building", "hostel-building.jpg", "SR University hostel building on campus", "campus hostel hostels building accommodation residence boys girls infrastructure"],
    ["hostel-room", "hostel-room.jpg", "Hostel room - clean, spacious rooms with platform beds and study tables", "hostel hostels room rooms accommodation interior stay dorm"],
    ["srix-building", "srix-building.jpg", "SRiX - SR Innovation Exchange (Technology Business Incubator) building", "campus srix incubator startup building innovation exchange infrastructure"],
    ["sports-basketball", "sports-basketball.jpg", "Basketball at SR University", "sports basketball ground court campus game playground games"],
    ["sports-basketball-2", "sports-basketball-2.jpg", "Basketball court - student match", "sports basketball court campus game playground games"],
    ["sports-cricket", "sports-cricket.jpg", "Cricket - SR Cricket Champions Trophy", "sports cricket ground campus game tournament trophy champions playground games"],
    ["placements-banner", "placements-banner.jpg", "Placement highlights - 1300+ placements, highest package 51 LPA", "placement placements highlights package offers banner jobs salary"],
    ["placements-51lpa", "placements-51lpa.jpg", "Beeraboina Shiva Krishna - 51 LPA, Microsoft (CSE 2025)", "placement placements microsoft 51lpa highest package shiva krishna student topper"],
    ["placements-2026", "placements-2026.jpg", "Batch 2026 top placements", "placement placements batch 2026 toppers students cisco amazon jpmorgan flipkart"],
    ["recruiters", "recruiters.jpg", "Recruiters who visit SR University", "placement placements recruiters companies recruiter logos"],
    ["industry-tieups", "industry-tieups.jpg", "Industrial tie-ups (National Instruments, IBM, Cisco and more)", "industry industrial tieups tie-ups partners companies collaboration mou logos"],
    ["internship-partners", "internship-partners.jpg", "Internship partners", "internship internships partners companies startups"],
    ["offers-1", "offers-1.jpg", "Placement offers by company (CTC)", "placement placements offers companies ctc package salary cisco infosys tcs ibm"],
    ["offers-2", "offers-2.jpg", "Placement offers by company (CTC) - continued", "placement placements offers companies ctc package salary cognizant accenture virtusa"],
    ["cgc", "cgc.jpg", "Career Guidance Cell (CGC)", "career guidance cell cgc placement counseling"],
    ["schools", "schools.jpg", "Schools at SR University", "school schools academics departments"],
    ["centers", "centers.jpg", "SRiX and Centers (Center for Design, AI & Deep Learning, Embedded Systems & IoT and more)", "centers center srix labs research incubator"],
    ["differentiators", "differentiators.jpg", "Key differentiators: Creative Engaging Faculty, Technology-Enabled Infrastructure, Collaborative Entrepreneurial Ecosystem", "about differentiators ecosystem faculty infrastructure"],
    ["regional-ecosystem", "regional-ecosystem.jpg", "Regional Ecosystem: Academic Excellence, Industry Relevance, Social Responsibility", "about vision mission ecosystem regional"],
    ["chancellor", "chancellor.jpg", "Sri A. Varada Reddy - Chancellor", "leadership leader people management team chancellor varada reddy"],
    ["pro-chancellor", "pro-chancellor.jpg", "Sri A. Madhukar Reddy - Pro Chancellor", "leadership leader people management team pro chancellor madhukar reddy"],
    ["vice-chancellor", "vice-chancellor.jpg", "Prof. Deepak Garg - Vice Chancellor", "leadership leader people management team vice chancellor vc deepak garg"],
    ["pro-vc", "pro-vc.jpg", "Dr. V. Mahesh - Pro Vice Chancellor", "leadership leader people management team pro vice chancellor mahesh"],
    ["registrar", "registrar.jpg", "Dr. P. Venkata Ramana Rao - Registrar", "leadership leader people management team registrar venkata ramana rao"],
    ["dean-ops", "dean-ops.jpg", "Dr. R. Archana Reddy - Dean, Operations & Growth", "leadership leader people management team dean operations growth archana reddy"],
    ["dean-planning", "dean-planning.jpg", "Dr. G. Shyamala - Dean, Planning", "leadership leader people management team dean planning shyamala"],
    ["dean-pg", "dean-pg.jpg", "Dr. J. Ajayan - Dean, PG Studies", "leadership leader people management team dean pg postgraduate studies ajayan"],
    ["dean-mentoring", "dean-mentoring.jpg", "Dr. Shivani Goel - Dean, Mentoring", "leadership leader people management team dean mentoring shivani goel"],
    ["dean-welfare", "dean-welfare.jpg", "Dr. AVV. Sudhakar - Dean, Students Welfare", "leadership leader people management team dean students welfare sudhakar"],
    ["dean-csai", "dean-csai.jpg", "Dr. Sheshikala Martha - Dean, School of Computer Science & AI", "leadership leader people management team dean computer science artificial intelligence cse ai sheshikala martha"],
    ["dean-engg", "dean-engg.jpg", "Dr. Md. Mujahid Irfan - Dean, School of Engineering", "leadership leader people management team dean engineering mujahid irfan"],
    ["dean-agri", "dean-agri.jpg", "Dr. G. Bhupal Raj - Dean, School of Agriculture", "leadership leader people management team dean agriculture agri bhupal raj"],
    ["dean-sciences", "dean-sciences.jpg", "Dr. Sujoy Saha - Dean, School of Sciences and Humanities", "leadership leader people management team dean sciences humanities sujoy saha"],
    ["srix-ceo", "srix-ceo.jpg", "Col B S Rao - CEO, SR Innovation Exchange (SRiX)", "leadership leader people management team ceo srix col rao"],
    ["iqac-director", "iqac-director.jpg", "Dr. P.V. Raja Shekar - Director, Internal Quality Assurance Cell", "leadership leader people management team director iqac quality assurance raja shekar"],
    ["eval-director", "eval-director.jpg", "Prof. G. Swamy Reddy - Director of Evaluation", "leadership leader people management team director evaluation exam swamy reddy"],
    ["admissions-director", "admissions-director.jpg", "Dr. Ram Raghotham Rao Deshmukh - Senior Director, Admissions and Marketing", "leadership leader people management team director admissions marketing ram raghotham deshmukh"],
    ["dean-academics", "dean-academics.jpg", "Dr. Srinivas Aluvala - Dean, Academics", "leadership leader people management team dean academics srinivas aluvala"],
    ["dean-innovation", "dean-innovation.jpg", "Dr. B. Girirajan - Dean, Innovation and Startups", "leadership leader people management team dean innovation startups girirajan"],
    ["adv-jayesh", "adv-jayesh.jpg", "Sri. Jayesh Ranjan - Principal Secretary, Industries & Commerce and IT Departments, Govt. of Telangana", "board advisors advisor jayesh ranjan"],
    ["adv-krishna", "adv-krishna.jpg", "Prof. Krishna Vedula - Professor & Dean Emeritus, UMass Lowell, USA", "board advisors advisor krishna vedula"],
    ["adv-tiwari", "adv-tiwari.jpg", "Dr. O.N. Tiwari - Principal Scientist, ICAR-IARI, New Delhi", "board advisors advisor tiwari"],
    ["adv-sridhar", "adv-sridhar.jpg", "Prof. Sridhar Condoor - Saint Louis University, USA", "board advisors advisor sridhar condoor"],
    ["adv-santhosh", "adv-santhosh.jpg", "Prof. Santhosh Pandey - Georgia Tech, USA", "board advisors advisor santhosh pandey"],
    ["adv-sanjeev", "adv-sanjeev.jpg", "Prof. Sanjeev Khanna - University of Missouri, Columbia", "board advisors advisor sanjeev khanna"],
    ["adv-rajesh", "adv-rajesh.jpg", "Prof. Rajesh Vasa - Deakin University, Australia", "board advisors advisor rajesh vasa"],
    ["adv-haim", "adv-haim.jpg", "Dr. Haim Levkowitz - UMass Lowell, USA", "board advisors advisor haim levkowitz"],
    ["adv-grc", "adv-grc.jpg", "Prof. G R C Reddy - Ex-Vice-Chancellor of SR University", "board advisors advisor grc reddy ex vice chancellor"],
    ["rank-the-impact", "rank-the-impact.jpg", "Times Higher Education Sustainability Impact Ratings 2026", "awards recognition rankings ranking times higher education sustainability impact the"],
    ["rank-the-asia", "rank-the-asia.jpg", "THE World University Rankings 2026 Asia: #351-400", "awards recognition rankings ranking times higher education the world asia"],
    ["rank-nirf", "rank-nirf.jpg", "NIRF Ranking 2025: 91 in Engineering, 101-150 in University category", "awards recognition rankings ranking nirf"],
    ["rank-sdg", "rank-sdg.jpg", "SDG rankings (THE Impact Rankings)", "awards recognition rankings ranking sdg"],
    ["rank-sdg-2", "rank-sdg-2.jpg", "Ranked 2 in India - Decent Work and Economic Growth (SDG 8)", "awards recognition rankings ranking sdg"],
    ["rank-the-private", "rank-the-private.jpg", "THE World University Rankings 2026: #7 private university in India, #801-1000 globally", "awards recognition rankings ranking the times higher education"],
    ["award-ariia", "award-ariia.jpg", "ARIIA - Atal Ranking of Institutions on Innovation Achievements", "awards recognition ariia atal ranking innovation"],
    ["award-icar", "award-icar.jpg", "First private university in Telangana & Andhra Pradesh with ICAR accreditation for School of Agriculture and B.Sc. (Hons.) Agriculture", "awards recognition icar accreditation agriculture"],
    ["award-nba", "award-nba.jpg", "NBA Tier-I Accreditation to all B.Tech programs", "awards recognition nba accreditation tier"],
    ["award-hma", "award-hma.jpg", "Academician Excellence Award 2023-24 - Hyderabad Management Association", "awards recognition award academician excellence hyderabad management association suman naredla"],
    ["award-incubation", "award-incubation.jpg", "Effective Practices of Incubation Center by Engineering Institute award, World Education Summit", "awards recognition award incubation center world education summit"],
    ["award-chevening", "award-chevening.jpg", "The Chevening Fellowship - Ms. Sreedevi Devireddy", "awards recognition award chevening fellowship sreedevi devireddy"],
    ["award-pride-telangana", "award-pride-telangana.jpg", "The Pride of Telangana Award 2019", "awards recognition award pride telangana sreedevi devireddy"],
    ["award-iucee-epics", "award-iucee-epics.jpg", "IUCEE-EPICS Consortium Award for contributions in Social Innovation", "awards recognition award iucee epics social innovation"],
    ["award-iucee-showcase", "award-iucee-showcase.jpg", "IUCEE Showcase Award at ICTIEE 2020", "awards recognition award iucee showcase ictiee"],
    ["award-national-entrepreneurship", "award-national-entrepreneurship.jpg", "National Entrepreneurship Award 2019 - Mentor of the year", "awards recognition award national entrepreneurship sreedevi devireddy"],
    ["award-education-leadership", "award-education-leadership.jpg", "Education Leadership Award - Business School Affaire and Dewang Mehta National Education Awards", "awards recognition award education leadership dewang mehta"],
    ["award-startup-india", "award-startup-india.jpg", "3 start-up ideas in the top 10 of Startup India initiative (Telangana)", "awards recognition award startup india telangana"],
    ["award-women-education", "award-women-education.jpg", "Women in Education Award - Dr. R. Archana Reddy", "awards recognition award women education archana reddy dewang mehta"],
    ["award-nari-udyamita", "award-nari-udyamita.jpg", "Nari Udyamita Award - Ms. Sreedevi Devireddy", "awards recognition award nari udyamita women entrepreneurship"],
    ["award-mic", "award-mic.jpg", "AICTE Chairman appreciating SR student start-up at MIC Innovation Festival", "awards recognition award aicte mic innovation festival startup"],
    ["award-hackathon", "award-hackathon.jpg", "AI Hackathon (TATA Motors) second runner-up", "awards recognition award hackathon tata motors ai"],
    ["award-sih", "award-sih.jpg", "Smart India Hackathon (SIH) 2019 - one of 19 nodal centers", "awards recognition award smart india hackathon sih nodal"]
]
_STOP = set("show me give send of the a an and for in at to is are was what does do how can you i want see view look looks like "
            "image images photo photos picture pictures pic pics gallery snap snaps please about sru sr university campus's "
            "some any all more other with from on my your our their its it this that these those them also too need get provide "
            "college colleges university universities campus anything everything".split())
_STRONG = re.compile(r"\b(image|images|photo|photos|picture|pictures|pic|pics|gallery|snap|snaps|visual|visuals|imgs?)\b", re.I)
_WEAK = re.compile(r"\b(show|see|view|look|looks|display)\b", re.I)

_FEATURED_IMAGE_IDS = (
    "hostel-building",
    "hostel-room",
    "srix-building",
    "sports-basketball",
    "sports-cricket",
    "placements-banner",
    "schools",
    "centers",
    "differentiators",
    "regional-ecosystem",
)

_IMAGE_STOP_WORDS = _STOP | {"sru", "sr", "university", "warangal", "picture", "pictures", "photo", "photos"}
_IMAGE_SYNONYMS = {
    "hostels": "hostel",
    "dining": "mess",
    "dinning": "mess",
    "dininghall": "mess",
    "dinninghall": "mess",
    "blocks": "block",
    "images": "image",
    "photos": "photo",
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_logger = logging.getLogger(__name__)


class _BingImageParser(HTMLParser):
    """Collects image result metadata from Bing Images result cards."""

    def __init__(self):
        super().__init__()
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributes = dict(attrs)
        if "iusc" not in (attributes.get("class") or "").split():
            return
        try:
            result = json.loads(attributes.get("m") or "{}")
        except json.JSONDecodeError:
            return
        image_url = result.get("turl") or result.get("murl")
        if not _is_https_url(image_url):
            image_url = result.get("murl")
        if not _is_https_url(image_url):
            return
        source_url = result.get("purl")
        self.results.append({
            "url": image_url,
            "source_url": source_url if _is_https_url(source_url) else image_url,
            "caption": re.sub(r"<[^>]*>", "", result.get("t") or "SR University image").strip(),
        })


class _SruImageParser(HTMLParser):
    """Collects relevant image assets directly from the SR University website."""

    def __init__(self, query_terms: set):
        super().__init__()
        self.query_terms = query_terms
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        attributes = dict(attrs)
        image_url = urljoin("https://sru.edu.in/", attributes.get("src") or "")
        if not _is_https_url(image_url) or urlparse(image_url).hostname != "sru.edu.in":
            return
        caption = attributes.get("alt") or attributes.get("title") or ""
        matched_terms = self.query_terms & _image_terms(f"{caption} {image_url}")
        if matched_terms:
            self.results.append({
                "url": image_url,
                "source_url": "https://sru.edu.in/",
                "caption": caption.strip() or "SR University campus image",
            })


def _is_https_url(value: str) -> bool:
    return isinstance(value, str) and urlparse(value).scheme == "https" and bool(urlparse(value).netloc)


def _image_terms(query: str) -> set:
    return {
        _IMAGE_SYNONYMS.get(_stem(word), _stem(word))
        for word in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if word not in _IMAGE_STOP_WORDS
    }


def _search_online_images(query: str, limit: int) -> list:
    """Search SR University's site, then relevant Bing Images results."""
    query_terms = _image_terms(query)
    if query_terms:
        request = Request(
            "https://sru.edu.in/",
            headers={"User-Agent": "Mozilla/5.0 (compatible; SRUniversityCampusAssistant/1.0)"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                page = response.read(2_000_000).decode("utf-8", errors="replace")
            parser = _SruImageParser(query_terms)
            parser.feed(page)
            if parser.results:
                return parser.results[:limit]
        except (HTTPException, URLError, TimeoutError, OSError, ValueError) as error:
            _logger.info("SR University site image lookup failed: %s", error)

    words = sorted(query_terms)[:6]
    subject = " ".join(words) or "campus"
    search_url = "https://www.bing.com/images/search?q=" + quote_plus(
        f"SR University Warangal {subject}"
    )
    request = Request(
        search_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SRUniversityCampusAssistant/1.0)"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            page = response.read(2_000_000).decode("utf-8", errors="replace")
        parser = _BingImageParser()
        parser.feed(page)
        relevant = []
        for image in parser.results:
            source_host = urlparse(image["source_url"]).hostname or ""
            image_terms = _image_terms(f"{image['caption']} {image['source_url']}")
            is_sru_result = (
                source_host == "sru.edu.in"
                or source_host.endswith(".sru.edu.in")
                or source_host == "srix.in"
                or source_host.endswith(".srix.in")
                or "sr university" in image["caption"].lower()
                or "warangal" in image["caption"].lower()
            )
            if is_sru_result and (not query_terms or query_terms & image_terms):
                relevant.append(image)
        return relevant[:limit]
    except (HTTPException, URLError, TimeoutError, OSError, ValueError) as error:
        _logger.warning("Online SR University image search failed: %s", error)
        return []


def _stem(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def pick_images(query: str, limit: int = 12) -> list:
    """Return matching local images first, searching online if the local library has none."""
    strong, weak = bool(_STRONG.search(query)), bool(_WEAK.search(query))
    if not (strong or weak):
        return []

    words = _image_terms(query)
    catalog_by_id = {item[0]: item for item in IMAGE_CATALOG}
    if not words:
        featured = [
            {"url": f"/images/{catalog_by_id[image_id][1]}", "caption": catalog_by_id[image_id][2]}
            for image_id in _FEATURED_IMAGE_IDS
            if image_id in catalog_by_id and os.path.isfile(os.path.join(IMAGE_DIR, catalog_by_id[image_id][1]))
        ][:limit]
        return featured or _search_online_images(query, limit)

    # Prefer files whose names directly match the requested subject.
    local_matches = []
    if os.path.isdir(IMAGE_DIR):
        for filename in os.listdir(IMAGE_DIR):
            path = os.path.join(IMAGE_DIR, filename)
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in _IMAGE_EXTENSIONS or not os.path.isfile(path):
                continue
            file_terms = _image_terms(stem.replace("-", " ").replace("_", " "))
            score = len(words & file_terms)
            if score:
                local_matches.append((score, filename))
    if local_matches:
        local_matches.sort(key=lambda match: (-match[0], match[1].lower()))
        return [
            {
                "url": f"/images/{filename}",
                "caption": os.path.splitext(filename)[0].replace("-", " ").replace("_", " ").title(),
            }
            for _, filename in local_matches[:limit]
        ]

    scored = []
    for i, (cid, fn, cap, tags) in enumerate(IMAGE_CATALOG):
        if not os.path.isfile(os.path.join(IMAGE_DIR, fn)):
            continue
        tagset = _image_terms(tags)
        score = len(words & tagset)
        role = cap.split(" - ")[-1].lower()          # e.g. "chancellor", "vice chancellor"
        if score and role in query.lower():          # exact role asked for -> rank it first
            score += len(role.split())
        if score:
            scored.append((score, -i, cid, fn, cap))
    if not scored:
        return _search_online_images(query, limit)
    scored.sort(reverse=True)
    top = scored[0][0]
    keep = [s for s in scored if s[0] == top] if top > 1 else scored
    keep.sort(key=lambda s: -s[1])  # catalogue order
    local_images = [{"url": f"/images/{fn}", "caption": cap} for _, _, _, fn, cap in keep[:limit]]
    return local_images or _search_online_images(query, limit)


# ----------------------------
# Step 3 - Nodes generation
# ----------------------------
def classifier_node(state: State) -> dict:
    """Labels the query (shown as a badge in the UI). Retrieval always searches BOTH PDFs."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about the university, programmes, courses, eligibility, "
        "specializations, schools, centers, faculty, leadership, placements, hostels, sports, "
        "awards, attendance, exams, grading, credits, or degree requirements.\n"
        "Use 'fee' for questions about tuition, fees, payment, refund, fellowships, stipends, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything else.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    category = llm.invoke(prompt).content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def retrieve_node(state: State) -> dict:
    """Searches BOTH PDFs for every question and merges the best chunks."""
    human = [m.content for m in state["messages"] if getattr(m, "type", "") == "human"]
    query = human[-1]
    # short follow-ups ("what about fees?") are searched together with the previous question
    if len(query.split()) < 6 and len(human) > 1:
        query = human[-2] + " " + query

    seen, parts = set(), []
    for name, retriever in (("Academic Handbook", acedemic_retriever), ("Fee Structure", fee_retriever)):
        for doc in retriever.invoke(query):
            text = doc.page_content.strip()
            if text in seen:
                continue
            seen.add(text)
            parts.append(f"[{name}]\n{text}")
    return {"retrieved_context": "\n\n".join(parts)}


def response_node(state: State) -> dict:
    """Generates a detailed answer from the PDFs, topped up with the LLM's own knowledge."""
    msgs = state["messages"]
    query = msgs[-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    history = "\n".join(
        f"{'Student' if m.type == 'human' else 'Assistant'}: {m.content}" for m in msgs[:-1][-6:]
    ) or "(none)"

    shown = state.get("images_shown", "")
    if shown == "NO_MATCH":
        image_rule = ("8. The student asked for images but none in the image library match. Do not say you cannot show "
                      "images; describe the topic vividly in words using the excerpts and your knowledge.\n")
    elif shown:
        image_rule = ("8. The app is displaying these images under your answer: " + shown + ". Never say you cannot show "
                      "images. Write a helpful answer describing what the images show and the related details.\n")
    else:
        image_rule = ""

    prompt = (
        f"You are the SR University (Warangal) assistant talking to a student of: {programme} "
        f"(format: level > programme > specialization).\n\n"
        "HOW TO ANSWER:\n"
        "1. Answer the question fully and in detail. Start from the document excerpts below.\n"
        "2. If the excerpts cover only part of the question, complete the answer with your own "
        "knowledge of SR University and the topic, so the student gets all the details.\n"
        "3. NEVER reply by telling the student to contact the university, visit the website, or "
        "check elsewhere. Always give the actual information yourself.\n"
        "4. Do not invent exact figures (fees, packages, dates, ranks, percentages). Use numbers "
        "from the excerpts; if an exact figure is not available, say it is not published in the "
        "available documents and explain everything else you do know about the topic.\n"
        "5. If the excerpts and your own knowledge differ, trust the excerpts.\n"
        "6. Tailor the answer to the student's programme where it is relevant.\n"
        "7. Format with **bold** headings, bullet lists and markdown tables where helpful "
        "(do not use # headings). For greetings or casual chat, reply naturally and briefly "
        "mention what you can help with.\n"
        + image_rule + "\n"
        f"Conversation so far:\n{history}\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Detailed answer:"
    )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# ----------------------------
# Step 5 - Building the graph
# ----------------------------
def build_graph():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "retrieve")
    graph.add_edge("retrieve", "response")
    graph.add_edge("response", END)

    return graph.compile()


app_graph = build_graph()


# ----------------------------
# Step 7 - Frontend page (embedded, so only ONE file is needed)
# ----------------------------
HTML_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>College Assistant</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎓</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg1:#1b1037; --bg2:#0d1b3a; --glass:rgba(255,255,255,.07); --edge:rgba(255,255,255,.16);
  --text:#f1ecff; --mute:#a99fcc; --pink:#ff6fae; --violet:#8b6bff;
  --academic:#6ab7ff; --fee:#ffbf5e; --general:#6fe3ae;
  box-sizing:border-box;
}
*,*::before,*::after{box-sizing:inherit}
html,body{height:100%;margin:0}
body{font-family:Figtree,system-ui,sans-serif;color:var(--text);background:linear-gradient(135deg,var(--bg1),var(--bg2));overflow:hidden;
  padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
h1,h2{font-family:'Bricolage Grotesque',Figtree,sans-serif;margin:0}
 
/* ---------- floating 3D background ---------- */
.bg{position:fixed;inset:0;perspective:900px;pointer-events:none;overflow:hidden}
.layer{position:absolute;inset:0;transition:transform .25s ease-out}
.orb{position:absolute;border-radius:50%;filter:blur(60px);opacity:.55;animation:drift 18s ease-in-out infinite alternate}
.o1{width:380px;height:380px;background:#7b4dff;top:-90px;left:-80px}
.o2{width:320px;height:320px;background:#ff5fa2;bottom:-80px;right:8%;animation-delay:-6s}
.o3{width:260px;height:260px;background:#27c6ff;top:38%;left:42%;opacity:.3;animation-delay:-11s}
@keyframes drift{to{transform:translate3d(60px,-50px,0) scale(1.15)}}
.shape{position:absolute;border:2px solid rgba(255,255,255,.22);transform-style:preserve-3d;animation:spin 16s linear infinite}
.s1{width:90px;height:90px;border-radius:50%;top:14%;right:9%}
.s2{width:60px;height:60px;border-radius:14px;bottom:18%;left:6%;animation-duration:22s;animation-direction:reverse}
.s3{width:130px;height:130px;border-radius:50%;bottom:8%;right:34%;animation-duration:28s;border-style:dashed}
.s4{width:44px;height:44px;border-radius:10px;top:10%;left:36%;animation-duration:19s;border-color:rgba(255,111,174,.5)}
@keyframes spin{from{transform:rotateX(0) rotateY(0) translateY(0)}50%{transform:rotateX(180deg) rotateY(120deg) translateY(-26px)}to{transform:rotateX(360deg) rotateY(360deg) translateY(0)}}
 
/* ---------- layout ---------- */
.app{position:relative;z-index:1;display:grid;grid-template-columns:340px 1fr;gap:28px;max-width:1180px;margin:0 auto;height:100dvh;padding:28px}
aside{display:flex;flex-direction:column;gap:22px;min-height:0}
.brand{animation:rise .9s cubic-bezier(.2,.9,.3,1.2) both}
.brand h1{font-size:2.5rem;line-height:1;font-weight:800;letter-spacing:-.02em}
.brand h1 span{display:inline-block;background:linear-gradient(90deg,var(--pink),var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent}
.brand p{color:var(--mute);margin:.6rem 0 0;line-height:1.45}
.logo{display:block;height:44px;width:auto;background:#fff;padding:7px 14px;border-radius:14px;margin-bottom:16px;box-shadow:0 10px 30px rgba(0,0,0,.3)}
@keyframes rise{from{opacity:0;transform:translateY(30px) rotateX(-25deg)}}
 
/* 3D programme carousel */
.pick h2{font-size:1rem;font-weight:500;color:var(--mute);margin-bottom:6px}
.carousel{position:relative;height:210px;perspective:900px;animation:bob 6s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-8px)}}
.card{position:absolute;left:50%;top:10px;width:158px;height:184px;margin-left:-79px;padding:18px 16px;border-radius:22px;cursor:pointer;
  background:linear-gradient(160deg,rgba(255,255,255,.18),rgba(255,255,255,.04));border:1px solid var(--edge);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  display:flex;flex-direction:column;justify-content:space-between;color:var(--text);font:inherit;text-align:left;
  transform:translateX(calc(var(--o)*120px)) translateZ(calc(var(--a)*-130px)) rotateY(calc(var(--o)*-38deg));
  opacity:calc(1 - var(--a)*.45);transition:transform .7s cubic-bezier(.2,.9,.25,1.1),opacity .5s,box-shadow .5s;z-index:calc(3 - var(--a));animation:fadein .45s both}
@keyframes fadein{from{opacity:0}}
.card.hide{opacity:0;pointer-events:none}
.card .ab{font-family:'Bricolage Grotesque';font-weight:800;font-size:2rem;letter-spacing:-.02em}
.card .ab.long{font-size:1.1rem;line-height:1.15}
.card small{color:var(--mute);line-height:1.35;font-size:.85rem;overflow:hidden}
.card.on{box-shadow:0 20px 50px rgba(139,107,255,.45),inset 0 0 0 1.5px var(--pink)}
.card:focus-visible{outline:3px solid #fff;outline-offset:3px}
.nav{display:flex;gap:8px;align-items:center;margin-top:2px}
.mini{background:var(--glass);border:1px solid var(--edge);color:var(--text);padding:6px 13px;border-radius:12px;font:inherit;font-size:.85rem;cursor:pointer;transition:background .25s}
.mini:hover:not(:disabled){background:rgba(255,255,255,.14)}
.mini:disabled{opacity:.4;cursor:default}
.hint{color:var(--mute);font-size:.78rem;margin-left:auto}
.status{color:var(--mute);font-size:.88rem;margin-top:6px}
.status b{color:var(--text)}
.legend{display:flex;flex-direction:column;gap:8px;font-size:.88rem;color:var(--mute)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:9px;box-shadow:0 0 12px currentColor}
.ghost{margin-top:auto;align-self:flex-start;background:var(--glass);border:1px solid var(--edge);color:var(--text);padding:10px 16px;border-radius:14px;font:inherit;cursor:pointer;transition:transform .25s,background .25s}
.ghost:hover{transform:translateY(-3px) rotateX(12deg);background:rgba(255,255,255,.14)}
 
/* ---------- chat panel ---------- */
.panel{display:flex;flex-direction:column;min-height:0;border-radius:28px;background:var(--glass);border:1px solid var(--edge);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
  box-shadow:0 30px 80px rgba(0,0,0,.4);animation:slide 1s cubic-bezier(.2,.9,.25,1) .15s both;transform-origin:left center}
@keyframes slide{from{opacity:0;transform:perspective(1200px) rotateY(-24deg) translateX(80px)}}
.feed{flex:1;overflow-y:auto;padding:26px 26px 10px;display:flex;flex-direction:column;gap:16px;perspective:900px;scroll-behavior:smooth}
.feed::-webkit-scrollbar{width:8px}.feed::-webkit-scrollbar-thumb{background:var(--edge);border-radius:8px}
.msg{max-width:82%;padding:13px 17px;border-radius:20px;line-height:1.55;font-size:.97rem;word-wrap:break-word;overflow-wrap:anywhere}
.msg.user{align-self:flex-end;background:linear-gradient(135deg,var(--violet),var(--pink));border-bottom-right-radius:6px;animation:inR .55s cubic-bezier(.2,.9,.3,1.15) both;transform-origin:right bottom}
.msg.ai{align-self:flex-start;background:rgba(255,255,255,.1);border:1px solid var(--edge);border-bottom-left-radius:6px;animation:inL .6s cubic-bezier(.2,.9,.3,1.15) both;transform-origin:left bottom}
@keyframes inR{from{opacity:0;transform:translateX(60px) rotateY(35deg) scale(.9)}}
@keyframes inL{from{opacity:0;transform:translateX(-60px) rotateY(-35deg) scale(.9)}}
.msg p{margin:0 0 .55em}.msg p:last-child{margin:0}
.msg ul,.msg ol{margin:.3em 0 .6em;padding-left:1.3em}
.msg code{background:rgba(0,0,0,.3);padding:1px 6px;border-radius:6px}
.tbl{overflow-x:auto;margin:.4em 0}
.msg table{border-collapse:collapse;font-size:.88rem}
.msg th,.msg td{border:1px solid var(--edge);padding:6px 10px;text-align:left}
.msg th{background:rgba(255,255,255,.08)}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:.74rem;font-weight:600;padding:3px 11px;border-radius:999px;margin-bottom:8px;border:1px solid currentColor}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:10px}
.gallery.one{grid-template-columns:minmax(0,420px)}
.gallery figure{margin:0}
.gallery img{width:100%;display:block;border-radius:12px;border:1px solid var(--edge);background:#fff;object-fit:contain;max-height:320px;cursor:zoom-in}
.gallery figcaption{font-size:.76rem;color:var(--mute);margin-top:4px;line-height:1.3}
.gallery figcaption a{display:inline-block;margin-left:6px;color:#a32632;font-weight:600}
.b-academic{color:var(--academic)}.b-fee{color:var(--fee)}.b-general{color:var(--general)}
.typing{align-self:flex-start;display:flex;gap:7px;padding:16px 20px;background:rgba(255,255,255,.1);border:1px solid var(--edge);border-radius:20px;animation:inL .4s both}
.typing span{width:9px;height:9px;border-radius:50%;background:var(--pink);animation:hop 1s ease-in-out infinite}
.typing span:nth-child(2){animation-delay:.15s;background:var(--violet)}.typing span:nth-child(3){animation-delay:.3s;background:var(--academic)}
@keyframes hop{50%{transform:translateY(-10px) scale(1.2)}}
 
/* welcome */
.welcome{margin:auto;text-align:center;padding:10px;perspective:800px}
.welcome h2{font-size:1.7rem;margin-bottom:6px}
.welcome p{color:var(--mute);margin:0 0 22px}
.chips{display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
.chip{border:1px solid currentColor;background:rgba(255,255,255,.06);padding:11px 18px;border-radius:16px;font:inherit;cursor:pointer;color:var(--c);animation:float 5s ease-in-out infinite;animation-delay:var(--d);transition:background .25s,box-shadow .25s}
.chip:hover{background:rgba(255,255,255,.14);box-shadow:0 12px 30px -8px currentColor}
@keyframes float{50%{transform:translateY(-12px) rotateX(10deg)}}
 
/* input */
.bar{display:flex;gap:12px;padding:16px 20px 20px}
.bar input{flex:1;min-width:0;background:rgba(0,0,0,.28);border:1px solid var(--edge);color:var(--text);padding:15px 20px;border-radius:18px;font:inherit;font-size:1rem;transition:box-shadow .3s,border-color .3s}
.bar input::placeholder{color:var(--mute)}
.bar input:focus{outline:none;border-color:var(--pink);box-shadow:0 0 0 4px rgba(255,111,174,.2)}
.send{width:54px;border:0;border-radius:18px;cursor:pointer;color:#fff;font-size:1.3rem;background:linear-gradient(135deg,var(--violet),var(--pink));transition:transform .3s cubic-bezier(.2,.9,.3,1.6),box-shadow .3s;box-shadow:0 10px 24px -6px var(--pink)}
.send:hover:not(:disabled){transform:translateY(-4px) rotate(-8deg) scale(1.06)}
.send:disabled{opacity:.5;cursor:wait}
button:focus-visible,input:focus-visible{outline:3px solid #fff;outline-offset:2px}
 
@media (max-width:860px){
  body{overflow:auto}
  .app{grid-template-columns:1fr;grid-template-rows:auto 1fr;gap:14px;padding:14px;height:100dvh}
  .brand h1{font-size:1.8rem}.brand p,.legend{display:none}
  .logo{height:30px;padding:5px 10px;margin-bottom:8px}
  .carousel{height:150px}.card{width:112px;height:128px;margin-left:-56px;padding:12px;top:8px;transform:translateX(calc(var(--o)*84px)) translateZ(calc(var(--a)*-90px)) rotateY(calc(var(--o)*-38deg))}
  .card .ab{font-size:1.3rem}.card .ab.long{font-size:.9rem}.card small{font-size:.7rem}
  .ghost{position:fixed;top:calc(14px + env(safe-area-inset-top,0px));right:14px;margin:0;padding:7px 12px;z-index:5}
  .msg{max-width:92%}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}

/* SR University portal */
:root{
  --bg1:#f3f6fa;--bg2:#f3f6fa;--glass:#fff;--edge:#e2e8f0;
  --text:#17263c;--mute:#65758b;--pink:#a32632;--violet:#193b69;
  --academic:#2768a8;--fee:#b07c19;--general:#237a58;
}
html{scroll-behavior:smooth}
body{height:auto;min-height:100%;overflow-x:hidden;overflow-y:auto;background:#f3f6fa;color:var(--text);padding:0}
.bg{display:none}
.site-header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #e6ebf1;box-shadow:0 5px 20px rgba(22,42,69,.05)}
.site-head-inner{max-width:1240px;min-height:82px;margin:auto;padding:10px 28px;display:flex;align-items:center;gap:28px}
.site-brand{display:flex;align-items:center;gap:13px;color:#16365f;text-decoration:none}
.site-logo-slot{display:flex;align-items:center;justify-content:center}
.site-brand .logo{height:54px;width:118px;object-fit:contain;padding:3px;background:#fff;border-radius:0;margin:0;box-shadow:none}
.site-wordmark{display:flex;flex-direction:column;gap:3px}
.site-wordmark strong{font-family:'Bricolage Grotesque',Figtree,sans-serif;font-size:1.08rem;letter-spacing:.035em}
.site-wordmark small{font-size:.75rem;color:#697b91}
.site-nav{display:flex;gap:28px;margin-left:auto}
.site-nav a{color:#34465d;text-decoration:none;font-size:.9rem;font-weight:600}
.site-nav a:hover{color:#a32632}
.srix-link{display:inline-flex;align-items:center;gap:8px;background:#a32632;color:#fff;text-decoration:none;font-weight:700;font-size:.86rem;padding:11px 17px;border-radius:4px;white-space:nowrap}
.campus-hero{position:relative;isolation:isolate;min-height:380px;overflow:hidden;background:#142b49;color:#fff}
.campus-slides,.campus-slide{position:absolute;inset:0}
.campus-slide{opacity:0;transition:opacity .45s ease}
.campus-slide.active{opacity:1}
.campus-slide img{width:100%;height:100%;display:block;object-fit:cover;object-position:center 48%}
.campus-hero::after{content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(90deg,rgba(11,29,53,.88) 0%,rgba(11,29,53,.58) 45%,rgba(11,29,53,.12) 100%)}
.hero-content{position:relative;z-index:2;max-width:1184px;min-height:380px;margin:auto;padding:70px 32px 76px;display:flex;flex-direction:column;justify-content:center}
.hero-kicker{font-size:.76rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:#f2c879;margin:0 0 14px}
.hero-content h1{max-width:680px;font-size:clamp(2.1rem,5vw,3.65rem);line-height:1.06;letter-spacing:-.025em}
.hero-content p{max-width:550px;margin:16px 0 24px;color:rgba(255,255,255,.88);font-size:1.05rem;line-height:1.65}
.hero-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.hero-actions a{display:inline-flex;padding:12px 18px;background:#a32632;color:#fff;text-decoration:none;font-weight:700;border-radius:4px}
.hero-actions a.secondary{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.55)}
.campus-controls{position:absolute;z-index:3;right:max(28px,calc((100vw - 1184px)/2 + 30px));bottom:24px;display:flex;align-items:center;gap:9px}
.campus-control,.campus-dot{width:36px;height:36px;border:1px solid rgba(255,255,255,.75);border-radius:50%;background:rgba(10,28,50,.35);color:#fff;font:inherit;cursor:pointer}
.campus-control:hover,.campus-dot:hover{background:#a32632}
.campus-dot{width:9px;height:9px;padding:0;border:0;background:rgba(255,255,255,.55)}
.campus-dot.active{width:24px;border-radius:8px;background:#fff}
.portal-content{max-width:1184px;margin:36px auto 0;padding:0 24px}
.srix-feature{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(280px,.9fr);align-items:stretch;background:#fff;border:1px solid #e5eaf0;box-shadow:0 10px 32px rgba(26,48,77,.06)}
.srix-copy{padding:30px 34px}
.eyebrow{color:#a32632;font-size:.73rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
.srix-copy h2,.assistant-heading h2{margin:8px 0 10px;font-size:1.7rem;color:#17365d}
.srix-copy p{max-width:630px;margin:0;color:#5f6e81;line-height:1.7}
.srix-photo{min-height:220px;overflow:hidden}
.srix-photo img{width:100%;height:100%;min-height:220px;display:block;object-fit:cover}
.srix-meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px;color:#425870;font-size:.83rem;font-weight:600}
.srix-meta span::before{content:"";display:inline-block;width:7px;height:7px;margin:0 8px 1px 0;border-radius:50%;background:#b32635}
.assistant-heading{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:42px 0 16px}
.assistant-heading p{margin:0;color:#65758b}
.campus-map-section{margin-top:38px;scroll-margin-top:100px}
.campus-map-heading{margin-bottom:16px}
.campus-map-heading h2{margin:8px 0;color:#17365d;font-size:1.7rem}
.campus-map-heading p{margin:0;color:#65758b;line-height:1.6}
.campus-map-card{display:grid;grid-template-columns:minmax(280px,.82fr) minmax(0,1.18fr);background:#fff;border:1px solid #e5eaf0;box-shadow:0 10px 32px rgba(26,48,77,.06)}
.route-form{padding:26px}
.route-form label{display:block;margin:0 0 7px;color:#34465d;font-size:.86rem;font-weight:700}
.route-form select{display:block;width:100%;min-height:44px;margin-bottom:18px;padding:10px 12px;border:1px solid #d6e0eb;border-radius:4px;background:#fff;color:#24364c;font:inherit}
.route-form select:focus{outline:3px solid rgba(39,104,168,.18);border-color:#2768a8}
.route-swap{display:flex;align-items:center;justify-content:space-between;margin:-8px 0 14px;color:#65758b;font-size:.8rem}
.route-swap button{border:0;background:none;color:#a32632;font:inherit;font-weight:700;cursor:pointer}
.route-submit{width:100%;min-height:46px;border:0;border-radius:4px;background:#a32632;color:#fff;font:inherit;font-weight:700;cursor:pointer}
.route-submit:hover{background:#84212a}
.route-status{min-height:46px;margin:14px 0 0;color:#425870;font-size:.86rem;line-height:1.55}
.route-link{display:none;align-items:center;justify-content:center;gap:8px;width:100%;margin-top:4px;padding:12px;background:#173d6b;color:#fff;text-decoration:none;font-weight:700;border-radius:4px}
.route-link.visible{display:flex}
.route-note{margin:16px 0 0;padding-top:12px;border-top:1px solid #edf0f4;color:#788698;font-size:.76rem;line-height:1.55}
.campus-map-frame{min-height:420px;background:#e9eef3}
.campus-map-frame iframe{width:100%;height:100%;min-height:420px;border:0;display:block}
.app{position:relative;z-index:1;grid-template-columns:320px minmax(0,1fr);gap:20px;max-width:1184px;height:auto;min-height:590px;margin:0 auto 56px;padding:0 24px}
.app aside{padding:22px;background:#fff;border:1px solid #e5eaf0;box-shadow:0 10px 32px rgba(26,48,77,.06);gap:18px}
.app .brand{display:none}
.pick h2{color:#17365d;font-weight:700;margin-bottom:8px}
.carousel{animation:none}
.card{background:linear-gradient(160deg,#fff,#f4f7fb);border-color:#dce4ee;color:#17365d;backdrop-filter:none;-webkit-backdrop-filter:none}
.card small{color:#65758b}
.card.on{box-shadow:0 12px 32px rgba(25,59,105,.2),inset 0 0 0 1.5px #a32632}
.mini{background:#fff;border-color:#dce4ee;color:#17365d}
.mini:hover:not(:disabled){background:#f1f4f8}
.hint,.status,.legend{color:#64758a}
.status b{color:#17365d}
.ghost{margin-top:8px;background:#fff;border-color:#dce4ee;color:#17365d}
.panel{min-height:590px;border-radius:4px;background:#fff;border:1px solid #e5eaf0;backdrop-filter:none;-webkit-backdrop-filter:none;box-shadow:0 10px 32px rgba(26,48,77,.07);animation:none}
.feed{min-height:0;padding:24px 24px 12px;perspective:none}
.feed::-webkit-scrollbar-thumb{background:#c9d3df}
.msg.user{background:#173d6b}
.msg.ai{background:#f5f7fa;border-color:#e4e9ef;color:#26364c}
.msg code{background:#e5eaf0}
.msg th{background:#eef2f6}
.msg th,.msg td{border-color:#dce4ee}
.badge{margin-bottom:8px}
.badge.b-academic{color:#2768a8}.badge.b-fee{color:#9a6c16}.badge.b-general{color:#237a58}
.welcome h2{color:#17365d}
.welcome p{color:#65758b}
.chip{background:#fff;animation:none}
.chip:hover{background:#f2f5f8}
.bar{border-top:1px solid #edf0f4}
.bar input{background:#f7f9fb;border-color:#dce4ee;color:#24364c}
.bar input::placeholder{color:#8390a0}
.bar input:focus{border-color:#a32632;box-shadow:0 0 0 3px rgba(163,38,50,.12)}
.send{background:#a32632;box-shadow:none;border-radius:4px}
.send:hover:not(:disabled){box-shadow:0 8px 20px rgba(163,38,50,.24)}
@media(max-width:860px){
  body{overflow-x:hidden}
  .site-head-inner{min-height:70px;padding:8px 16px;gap:12px}
  .site-brand .logo{width:88px;height:46px}
  .site-wordmark{display:none}
  .site-nav{display:flex;gap:0;margin-left:auto}
  .site-nav a:not(.map-nav-link){display:none}
  .site-nav .map-nav-link{padding:9px 10px;border:1px solid #e2e8f0;border-radius:4px}
  .srix-link{margin-left:auto;padding:9px 12px}
  .campus-hero,.hero-content{min-height:340px}
  .hero-content{padding:48px 22px 76px}
  .hero-content h1{font-size:2.35rem}
  .campus-controls{right:18px;bottom:16px}
  .portal-content{margin-top:22px;padding:0 16px}
  .srix-feature{grid-template-columns:1fr}
  .srix-copy{padding:24px}
  .srix-photo{min-height:180px}
  .srix-photo img{min-height:180px;max-height:240px}
  .campus-map-section{margin-top:28px}
  .campus-map-card{grid-template-columns:1fr}
  .route-form{padding:20px}
  .campus-map-frame,.campus-map-frame iframe{min-height:320px}
  .assistant-heading{align-items:start;flex-direction:column;gap:5px;margin:30px 0 14px}
  .app{grid-template-columns:1fr;grid-template-rows:auto minmax(460px,1fr);gap:14px;height:auto;min-height:0;margin-bottom:30px;padding:0 16px}
  .app aside{padding:16px}
  .carousel{height:150px}
  .panel{min-height:500px}
  .feed{padding:18px 14px 10px}
  .bar{padding:12px}
  .bar input{padding:13px 14px}
  .send{width:48px}
}
</style>
</head>
<body>
<div class="bg" aria-hidden="true">
  <div class="layer" data-k="14"><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div></div>
  <div class="layer" data-k="34"><div class="shape s1"></div><div class="shape s2"></div><div class="shape s3"></div><div class="shape s4"></div></div>
</div>
<header class="site-header">
  <div class="site-head-inner">
    <a class="site-brand" href="#home" aria-label="SR University portal home">
      <span class="site-logo-slot"></span>
      <span class="site-wordmark"><strong>SR UNIVERSITY</strong><small>Student information portal</small></span>
    </a>
    <nav class="site-nav" aria-label="Main navigation">
      <a href="#home">Home</a>
      <a href="#srix">Innovation &amp; SRiX</a>
      <a class="map-nav-link" href="#campus-map" aria-label="Open campus maps and walking directions">Maps</a>
      <a href="#assistant">Ask SRU</a>
    </nav>
    <a class="srix-link" href="#srix">Explore SRiX <span aria-hidden="true">↗</span></a>
  </div>
</header>
<section class="campus-hero" id="home" aria-label="SR University campus">
  <div class="campus-slides" id="campusSlides" aria-live="off"></div>
  <div class="hero-content">
    <p class="hero-kicker">SR University · Warangal, Telangana</p>
    <h1>A campus where ideas become impact.</h1>
    <p>Explore student life, innovation, academics and the spaces that make SR University feel like home.</p>
    <div class="hero-actions">
      <a href="#assistant">Ask the SRU assistant</a>
      <a class="secondary" href="#srix">Discover SRiX</a>
    </div>
  </div>
  <div class="campus-controls" aria-label="Campus image slideshow controls">
    <button class="campus-control" id="campusPrev" type="button" aria-label="Previous campus image">‹</button>
    <div id="campusDots" aria-label="Choose campus image"></div>
    <button class="campus-control" id="campusNext" type="button" aria-label="Next campus image">›</button>
  </div>
</section>
<section class="portal-content" id="srix" aria-labelledby="srixTitle">
  <div class="srix-feature">
    <div class="srix-copy">
      <span class="eyebrow">Innovation · Entrepreneurship · Industry</span>
      <h2 id="srixTitle">Meet SRiX — ideas in motion</h2>
      <p>SR Innovation Exchange (SRiX) is SR University's technology business incubator, bringing students, founders and industry together to turn promising ideas into meaningful ventures.</p>
      <div class="srix-meta"><span>Startup incubation</span><span>Industry connections</span><span>Technology-led learning</span></div>
    </div>
    <div class="srix-photo"><img src="/images/srix-building.jpg" alt="SRiX, SR University's innovation exchange building" loading="lazy"></div>
  </div>
</section>
<section class="portal-content campus-map-section" id="campus-map" aria-labelledby="campusMapTitle">
  <div class="campus-map-heading">
    <span class="eyebrow">Find your way around</span>
    <h2 id="campusMapTitle">Campus map &amp; walking directions</h2>
    <p>Choose where you are and the block you’re heading to. Open the walking route in Google Maps for navigation.</p>
  </div>
  <div class="campus-map-card">
    <form class="route-form" id="routeForm">
      <label for="routeOrigin">Starting point</label>
      <select id="routeOrigin" name="origin">
        <option value="current">My current location</option>
        <option value="block1">Block 1</option>
        <option value="library">Library — Block 2</option>
        <option value="hostel">Hostels</option>
        <option value="mess">Mess / Dining</option>
        <option value="srix">SRiX building</option>
      </select>
      <div class="route-swap">
        <span>Walking route across campus</span>
        <button id="swapRoute" type="button">Swap points ↕</button>
      </div>
      <label for="routeDestination">Destination</label>
      <select id="routeDestination" name="destination">
        <option value="block1">Block 1</option>
        <option value="library">Library — Block 2</option>
        <option value="hostel">Hostels</option>
        <option value="mess">Mess / Dining</option>
        <option value="srix">SRiX building</option>
      </select>
      <button class="route-submit" type="submit">Get walking directions</button>
      <p class="route-status" id="routeStatus" role="status" aria-live="polite">Choose a starting point and destination to see the route.</p>
      <a class="route-link" id="routeLink" href="#" target="_blank" rel="noopener noreferrer">Open route in Google Maps <span aria-hidden="true">↗</span></a>
      <p class="route-note">Block names are based on the campus image labels available in this app. Google Maps supplies the route; indoor corridors and campus-only paths may not be mapped.</p>
    </form>
    <div class="campus-map-frame">
      <iframe title="Map of SR University, Ananthasagar campus" loading="lazy" referrerpolicy="no-referrer-when-downgrade" src="https://maps.google.com/maps?q=SR%20University%2C%20Ananthasagar%2C%20Hasanparthy%2C%20Warangal%2C%20Telangana%20506371&output=embed"></iframe>
    </div>
  </div>
</section>
<div class="portal-content" id="assistant">
  <div class="assistant-heading">
    <div><span class="eyebrow">Your campus, made easier</span><h2>How can we help?</h2></div>
    <p>Ask about programmes, fees, campus life and more.</p>
  </div>
<div class="app">
  <aside>
    <div class="brand">
      <img class="logo" alt="SR University" src="data:image/png;base64,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">
      <h1>🎓 College<br><span>Assistant</span></h1>
      <p>Ask about academics, fees, or anything campus-related.</p>
    </div>
    <div class="pick">
      <h2 id="ptitle">Choose your level</h2>
      <div class="carousel" id="car" role="radiogroup" aria-label="Programme"></div>
      <div class="nav">
        <button class="mini" id="back" type="button">← Back</button>
        <button class="mini" id="prev" type="button" aria-label="Previous">‹</button>
        <button class="mini" id="next" type="button" aria-label="Next">›</button>
        <span class="hint" id="hint"></span>
      </div>
      <div class="status" id="status"></div>
    </div>
    <div class="legend">
      <span><i style="color:var(--academic);background:currentColor"></i>Academic Handbook</span>
      <span><i style="color:var(--fee);background:currentColor"></i>Fee Structure</span>
      <span><i style="color:var(--general);background:currentColor"></i>General knowledge</span>
    </div>
    <button class="ghost" id="clear">🗑️ Clear chat</button>
  </aside>
 
  <main class="panel">
    <div class="feed" id="feed" aria-live="polite"></div>
    <div class="bar">
      <input id="q" type="text" placeholder="Type your question here..." autocomplete="off" aria-label="Your question">
      <button class="send" id="send" aria-label="Send">➤</button>
    </div>
  </main>
</div>
</div>
 
<script>
/* S("short|Full name", ...) -> list of {id,name} */
const S=(...a)=>a.map(x=>{const [i,n]=x.split("|");return {id:i,name:n||i}});
const SCHOOLS=[
  {id:"CS & AI",name:"School of Computer Science & Artificial Intelligence",specs:S("Computer Science and Engineering","Artificial intelligence","Mathematics","Statistics")},
  {id:"Engineering",name:"School of Engineering",specs:S("Electronics and Communication Engineering","Electrical and Electronics Engineering","Civil Engineering","Mechanical Engineering","Material Science","Sustainable Development","Disaster Management and Mitigation")},
  {id:"Business",name:"School of Business",specs:S("Management","Entrepreneurship")},
  {id:"Sciences",name:"School of Sciences and Humanities",specs:S("English","Physics","Chemistry","Cognitive Science","Liberal Arts","Indian Knowledge Systems","Linguistic science and Technology","Cultural Studies","Literary Studies","Biotechnology")},
  {id:"Agriculture",name:"School of Agriculture",specs:S("Agriculture","Agronomy","Horticulture","Genetics & Plant Breeding","Entomology","Plant Pathology","Soil Science","Agricultural Economics","Agricultural Extension")}
];
const TREE=[
  {id:"UG",name:"Undergraduate programmes",progs:[
    {id:"B.Tech",name:"Engineering",specs:S(
      "CSE|Computer Science and Engineering","CSE - AI & ML|CSE - Artificial Intelligence & Machine Learning","CSE - Cyber|CSE - Cybersecurity","CSE - Data Science|CSE - Data Science",
      "ECE|Electronics and Communication Engineering","ECE - AI & ML|ECE - Artificial Intelligence & Machine Learning","ECE - VLSI|ECE - VLSI",
      "EEE - Siemens|EEE - In partnership with Siemens","EEE - EV|EEE - Electric Vehicles","ME - Cyient|ME - In partnership with Cyient","ME - Robotics|ME - Robotics and Automation",
      "CE - DivyaSree|Civil Engineering - In partnership with DivyaSree NSL infrastructure")},
    {id:"BBA",name:"Business Administration",specs:S("BBA|Bachelor of Business Administration","Marketing|BBA - Marketing","Finance|BBA - Finance","Business Analytics|BBA - Business Analytics")},
    {id:"BCA",name:"Computer Applications"},
    {id:"B.Com",name:"Bachelor of Commerce"},
    {id:"B.Sc (Hons.)",name:"Honours degrees",specs:S("Agriculture|B.Sc. (Hons.) Agriculture","Computer Science|B.Sc. (Hons.) Computer Science")},
    {id:"B.Sc",name:"Science degrees",specs:S("Physics","Chemistry","Mathematics","Forensic Science")},
    {id:"B.Des",name:"Bachelor of Design",specs:S("Interior Design|Bachelor of Interior Design","Fashion Design|Bachelor of Fashion Design")},
    {id:"IEP",name:"International Engineering Program",specs:S("CSE|Computer Science and Engineering (with USA Universities)","ECE|Electronics and Communication Engineering (with USA Universities)")},
    {id:"IBAP",name:"International Business Accelerated Program (BBA + MBA/MS)",specs:S("Finance|MBA/MS in Finance","Business Analytics|MBA/MS in Business Analytics")}
  ]},
  {id:"PG",name:"Postgraduate programmes",progs:[
    {id:"M.Tech",name:"Master of Technology",specs:S(
      "CSE|M.Tech. (Computer Science and Engineering)","CSE - AI & ML|M.Tech. (CSE) - Artificial Intelligence & Machine Learning","ECE - VLSI|M.Tech. (ECE) - VLSI",
      "ECE - Embedded|M.Tech. (ECE) - Embedded Systems","ECE - EDT|M.Tech. (ECE) - Electronic Design Technology","EEE - Power|M.Tech. (EEE) - Power Electronics",
      "ME - Adv. Mfg|M.Tech. (ME) - Advanced Manufacturing Systems","CE - Construction|M.Tech. (CE) - Construction Technology & Management")},
    {id:"MBA",name:"Master of Business Administration"},
    {id:"MCA",name:"Master of Computer Applications"},
    {id:"M.Sc",name:"Master of Science",specs:S("Computer Science","Bio Chemistry","Statistics","Physics","Chemistry")},
    {id:"M.Sc (Agri)",name:"M.Sc. (Agriculture)",specs:S("Agronomy","Genetics and Plant Breeding","Horticulture","Agricultural Extension","Agricultural Entomology")},
    {id:"Accelerated MS",name:"Accelerated Masters Program",specs:S("MS - CS Tech|Accelarated MS in Computer Systems Technology","MS - Agribusiness|Masters in International Agribusiness Management","MS - Finance","MS - Public Admin|MS in Public Administration","Professional MBA","MS - Comp. Science|MS in Computer Science (University of Florida)")}
  ]},
  {id:"Ph.D",name:"Doctoral programmes",progs:SCHOOLS},
  {id:"Post-Doc",name:"Post-Doctoral programmes",progs:SCHOOLS}
];
const SUGGEST=[
  {t:"What is the eligibility for my programme?",c:"var(--academic)"},
  {t:"What fellowships and fees are listed?",c:"var(--fee)"},
  {t:"Hi! What can you help me with?",c:"var(--general)"}
];
let lvl=null,prog=null,spec=null,cur=0,history=[],busy=false,shown=null;
const $=s=>document.querySelector(s),feed=$("#feed"),q=$("#q"),sendBtn=$("#send"),car=$("#car");

const CAMPUS_ADDRESS="SR University, Ananthasagar, Hasanparthy, Warangal, Telangana 506371";
const CAMPUS_STOPS={
  block1:"Block 1",
  library:"Library, Block 2",
  hostel:"Hostels",
  mess:"Mess and Dining",
  srix:"SRiX building"
};
const originSelect=$("#routeOrigin"),destinationSelect=$("#routeDestination");
const routeStatus=$("#routeStatus"),routeLink=$("#routeLink");
$("#swapRoute").addEventListener("click",()=>{
  if(originSelect.value==="current"){
    routeStatus.textContent="Choose a named starting block before swapping the route.";
    return;
  }
  [originSelect.value,destinationSelect.value]=[destinationSelect.value,originSelect.value];
  routeLink.classList.remove("visible");
  routeStatus.textContent="Route points swapped. Get directions to update the route.";
});
$("#routeForm").addEventListener("submit",event=>{
  event.preventDefault();
  const destination=CAMPUS_STOPS[destinationSelect.value];
  if(!destination){
    routeStatus.textContent="Choose a campus destination to continue.";
    routeLink.classList.remove("visible");
    return;
  }
  if(originSelect.value!=="current"&&originSelect.value===destinationSelect.value){
    routeStatus.textContent="Choose two different campus locations.";
    routeLink.classList.remove("visible");
    return;
  }
  const openRoute=origin=>{
    const url=new URL("https://www.google.com/maps/dir/");
    url.searchParams.set("api","1");
    url.searchParams.set("origin",origin);
    url.searchParams.set("destination",`${destination}, ${CAMPUS_ADDRESS}`);
    url.searchParams.set("travelmode","walking");
    routeLink.href=url.toString();
    routeLink.classList.add("visible");
    routeStatus.textContent=`Walking route: ${originSelect.value==="current"?"Your current location":CAMPUS_STOPS[originSelect.value]+", SR University"} → ${destination}, SR University. Google Maps will show the available route and directions.`;
    routeLink.focus();
  };
  if(originSelect.value==="current"){
    if(!navigator.geolocation){
      routeStatus.textContent="This browser does not provide location access. Select a named starting block instead.";
      routeLink.classList.remove("visible");
      return;
    }
    routeStatus.textContent="Requesting your location…";
    navigator.geolocation.getCurrentPosition(
      position=>openRoute(`${position.coords.latitude},${position.coords.longitude}`),
      error=>{
        const message=error.code===error.PERMISSION_DENIED
          ?"Location access was denied. Allow location access or choose a named starting block."
          :"Could not get your location. Try again or choose a named starting block.";
        routeStatus.textContent=message;
        routeLink.classList.remove("visible");
      },
      {enableHighAccuracy:true,timeout:10000,maximumAge:60000}
    );
    return;
  }
  openRoute(`${CAMPUS_STOPS[originSelect.value]}, ${CAMPUS_ADDRESS}`);
});
originSelect.addEventListener("change",()=>routeLink.classList.remove("visible"));
destinationSelect.addEventListener("change",()=>routeLink.classList.remove("visible"));

const campusGallery=[
  ["campus-ranking-2025-enhanced.jpg","SR University campus and NIRF 2025 recognition"],
  ["campus-orientation-enhanced.jpg","SR University campus during the orientation programme"],
  ["campus-courtyard-enhanced.jpg","Campus courtyard and student gathering space"],
  ["campus-dining-enhanced.jpg","SR University dining space"],
  ["campus-hostel-spaces.avif","Student hostel living spaces"],
  ["campus-hostel-room-enhanced.jpg","Student hostel room and study space"],
  ["srix-building-user-enhanced.jpg","SRiX — SR Innovation Exchange"],
  ["campus-hostel-building-enhanced.jpg","SR University student residence"]
];
const slides=$("#campusSlides"),dots=$("#campusDots");
let campusIndex=0,campusTimer;
campusGallery.forEach(([file,caption],i)=>{
  const slide=document.createElement("div");
  slide.className="campus-slide"+(i===0?" active":"");
  slide.setAttribute("aria-hidden",String(i!==0));
  const image=document.createElement("img");
  image.src="/images/"+file;image.alt=caption;image.loading=i===0?"eager":"lazy";
  slide.appendChild(image);slides.appendChild(slide);
  const dot=document.createElement("button");
  dot.className="campus-dot"+(i===0?" active":"");dot.type="button";
  dot.setAttribute("aria-label","Show campus image "+(i+1));dot.setAttribute("aria-pressed",String(i===0));
  dot.addEventListener("click",()=>showCampusSlide(i,true));dots.appendChild(dot);
});
function showCampusSlide(index,resetTimer=false){
  const allSlides=[...slides.children],allDots=[...dots.children];
  campusIndex=(index+campusGallery.length)%campusGallery.length;
  allSlides.forEach((slide,i)=>{const active=i===campusIndex;slide.classList.toggle("active",active);slide.setAttribute("aria-hidden",String(!active))});
  allDots.forEach((dot,i)=>{const active=i===campusIndex;dot.classList.toggle("active",active);dot.setAttribute("aria-pressed",String(active))});
  if(resetTimer)startCampusSlideshow();
}
function startCampusSlideshow(){
  clearInterval(campusTimer);
  if(!matchMedia("(prefers-reduced-motion: reduce)").matches){
    campusTimer=setInterval(()=>showCampusSlide(campusIndex+1),1500);
  }
}
$("#campusPrev").addEventListener("click",()=>showCampusSlide(campusIndex-1,true));
$("#campusNext").addEventListener("click",()=>showCampusSlide(campusIndex+1,true));
$(".campus-hero").addEventListener("mouseenter",()=>clearInterval(campusTimer));
$(".campus-hero").addEventListener("mouseleave",startCampusSlideshow);
document.querySelector(".site-logo-slot").append(document.querySelector(".logo"));
startCampusSlideshow();
 
/* selection logic: level -> programme -> specialization */
const list=()=>!lvl?TREE:!prog||!prog.specs?lvl.progs:prog.specs;
const done=()=>!!(lvl&&prog&&(!prog.specs||spec));
const path=()=>[lvl,prog,spec].filter(Boolean).map(x=>x.id).join(" › ");
function choose(item){
  if(!lvl)lvl=item;
  else if(!prog||!prog.specs){prog=item;spec=null}
  else spec=item;
  cur=done()?Math.max(0,list().indexOf(item)):0;
  layout();
}
function go(i){ if(done())choose(list()[i]); else{cur=i;layout()} }
function back(){
  let popped;
  if(spec){popped=spec;spec=null}
  else if(prog){popped=prog;prog=null}
  else if(lvl){popped=lvl;lvl=null}
  else return;
  cur=Math.max(0,list().indexOf(popped));layout();
}
 
/* carousel */
function build(L){
  car.innerHTML="";
  L.forEach((p,i)=>{
    const b=document.createElement("button");
    b.className="card";b.type="button";b.setAttribute("role","radio");
    b.innerHTML=`<span class="ab${p.id.length>8?" long":""}">${esc(p.id)}</span><small>${esc(p.name)}</small>`;
    b.onclick=()=>{ if(done())go(i); else if(i!==cur){cur=i;layout()} else choose(L[i]) };
    car.appendChild(b);
  });
  shown=L;
}
function layout(){
  const L=list(),N=L.length;
  if(shown!==L)build(L);
  [...car.children].forEach((c,i)=>{
    let o=((i-cur)%N+N)%N; if(o>N/2)o-=N;
    const hide=Math.abs(o)>1; if(hide)o=o<0?-1:1;
    c.style.setProperty("--o",o);c.style.setProperty("--a",Math.abs(o));
    c.classList.toggle("hide",hide);
    c.classList.toggle("on",o===0&&!hide);c.setAttribute("aria-checked",i===cur);
  });
  $("#ptitle").textContent=!lvl?"Choose your level":!prog?"Choose your programme":(prog.specs&&!spec)?"Choose your specialization":"Your selection";
  $("#hint").textContent=done()?"Tap another card to change":"Tap the centre card";
  $("#back").disabled=!lvl;
  $("#status").innerHTML=lvl?`Answering as a <b>${esc(path())}</b> student`:"Pick your level, programme &amp; specialization";
}
$("#back").onclick=back;
$("#prev").onclick=()=>go((cur-1+list().length)%list().length);
$("#next").onclick=()=>go((cur+1)%list().length);
addEventListener("keydown",e=>{
  if(e.target===q)return;
  if(e.key==="ArrowLeft")$("#prev").click();
  if(e.key==="ArrowRight")$("#next").click();
});
 
/* mouse parallax */
addEventListener("mousemove",e=>{
  const x=e.clientX/innerWidth-.5,y=e.clientY/innerHeight-.5;
  document.querySelectorAll(".layer").forEach(l=>{const k=+l.dataset.k;l.style.transform=`translate3d(${-x*k}px,${-y*k}px,0)`});
});
 
/* tiny safe markdown */
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const inline=s=>esc(s).replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/`(.+?)`/g,"<code>$1</code>");
function md(text){
  const L=text.split("\n"),out=[];let i=0;
  while(i<L.length){
    const l=L[i].trim();
    if(!l){i++;continue}
    if(l.startsWith("|")){
      const rows=[];while(i<L.length&&L[i].trim().startsWith("|")){rows.push(L[i].trim());i++}
      const cells=r=>r.replace(/^\||\|$/g,"").split("|").map(c=>c.trim());
      const body=rows.filter(r=>!/^\|[\s:|-]+\|?$/.test(r));
      out.push('<div class="tbl"><table>'+body.map((r,k)=>"<tr>"+cells(r).map(c=>`<${k?"td":"th"}>${inline(c)}</${k?"td":"th"}>`).join("")+"</tr>").join("")+"</table></div>");
    }else if(/^([-*•]|\d+[.)])\s/.test(l)){
      const ord=/^\d/.test(l),items=[];
      while(i<L.length&&/^([-*•]|\d+[.)])\s/.test(L[i].trim())){items.push(L[i].trim().replace(/^([-*•]|\d+[.)])\s/,""));i++}
      out.push(`<${ord?"ol":"ul"}>`+items.map(x=>`<li>${inline(x)}</li>`).join("")+`</${ord?"ol":"ul"}>`);
    }else{
      const p=[];while(i<L.length&&L[i].trim()&&!L[i].trim().startsWith("|")&&!/^([-*•]|\d+[.)])\s/.test(L[i].trim())){p.push(inline(L[i].trim()));i++}
      out.push("<p>"+p.join("<br>")+"</p>");
    }
  }
  return out.join("");
}
 
/* UI helpers */
function welcome(){
  feed.innerHTML=`<div class="welcome"><h2>What do you want to know?</h2><p>Pick a question or type your own.</p><div class="chips"></div></div>`;
  const box=feed.querySelector(".chips");
  SUGGEST.forEach((s,i)=>{
    const b=document.createElement("button");b.className="chip";b.textContent=s.t;
    b.style.setProperty("--c",s.c);b.style.setProperty("--d",(i*-1.3)+"s");
    b.onclick=()=>ask(s.t);box.appendChild(b);
  });
}
function add(role,html,type){
  const w=feed.querySelector(".welcome");if(w)w.remove();
  const d=document.createElement("div");d.className="msg "+role;
  d.innerHTML=(type?`<span class="badge b-${type}">${type.toUpperCase()}</span>`:"")+html;
  feed.appendChild(d);feed.scrollTop=feed.scrollHeight;return d;
}
function gallery(imgs){
  if(!imgs||!imgs.length)return"";
  return `<div class="gallery${imgs.length===1?" one":""}">`+imgs.map(i=>`<figure><a href="${esc(i.source_url||i.url)}" target="_blank" rel="noopener"><img src="${esc(i.url)}" alt="${esc(i.caption)}" loading="lazy" onload="feed.scrollTop=feed.scrollHeight"></a><figcaption>${esc(i.caption)}${i.source_url?`<a href="${esc(i.source_url)}" target="_blank" rel="noopener">Source</a>`:""}</figcaption></figure>`).join("")+"</div>";
}
function typing(){
  const t=document.createElement("div");t.className="typing";t.innerHTML="<span></span><span></span><span></span>";
  feed.appendChild(t);feed.scrollTop=feed.scrollHeight;return t;
}
 
async function ask(text){
  text=text.trim();if(!text||busy)return;
  busy=true;sendBtn.disabled=true;q.value="";
  add("user",`<p>${esc(text)}</p>`);
  history.push({role:"human",content:text});
  const t=typing();
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({programme:path()||"Unknown",messages:history})});
    if(!r.ok)throw new Error("Server error "+r.status);
    const d=await r.json();
    t.remove();add("ai",md(d.reply)+gallery(d.images),d.query_type);
    history.push({role:"ai",content:d.reply});
  }catch(e){
    t.remove();history.pop();
    add("ai",`<p>Couldn't reach the assistant. Check your connection and send the question again.</p>`);
  }
  busy=false;sendBtn.disabled=false;q.focus();
}
 
sendBtn.onclick=()=>ask(q.value);
q.addEventListener("keydown",e=>{if(e.key==="Enter")ask(q.value)});
$("#clear").onclick=()=>{history=[];welcome()};
layout();
welcome();
</script>
</body>
</html>
'''
 
 
# ----------------------------
# Step 6 - Web API (replaces the Streamlit UI)
# ----------------------------
app = FastAPI(title="College Assistant")
 
 
class Turn(BaseModel):
    role: str       # "human" or "ai"
    content: str
 
 
class ChatRequest(BaseModel):
    programme: str         # e.g. "UG › B.Tech › CSE - AI & ML"
    messages: List[Turn]   # full history, last item = new user message
 
 
@app.post("/api/chat")
def chat(req: ChatRequest):
    lc_messages = [(t.role, t.content) for t in req.messages]
    last_user = next((t.content for t in reversed(req.messages) if t.role == "human"), "")
    images = pick_images(last_user)
    wanted = bool(_STRONG.search(last_user) or _WEAK.search(last_user))
    shown = "; ".join(i["caption"] for i in images) if images else ("NO_MATCH" if wanted else "")
    result = app_graph.invoke({
        "programme": req.programme,
        "messages": lc_messages,
        "images_shown": shown
    })
    ai_response = result["messages"][-1].content
    query_type = result.get("query_type", "general")
    return {"reply": ai_response, "query_type": query_type, "images": images}


if os.path.isdir(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE



#.\.venv-py313\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000