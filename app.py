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
from dotenv import load_dotenv
from bm25 import BM25Retriever
 
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
    def build_retriever(pdf_filename: str):
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_filename)
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
 
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(document)
        return BM25Retriever(chunks, k=8)
 
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
 
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
 
    return academic_retriever, fee_retriever, llm
 
 
academic_retriever, fee_retriever, llm = load_resources()
 
 
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
    for name, retriever in (("Academic Handbook", academic_retriever), ("Fee Structure", fee_retriever)):
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
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>New Web Service</title>
  <style>
    :root {
      --bg: #f2f4f7;
      --panel: #ffffff;
      --panel-2: #f7f8fa;
      --line: #dfe3e8;
      --line-strong: #c7ccd3;
      --text: #1d232b;
      --muted: #677583;
      --accent: #7a52d4;
      --accent-soft: #f3ebff;
      --button: #1d1d1f;
      --button-hover: #121416;
      --danger: #d85050;
      --success: #2f6d33;
      --purple: #a36af3;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    body {
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 16px;
    }

    .page {
      width: 980px;
      background: var(--bg);
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 18px 12px 10px;
    }

    .title {
      margin: 0 0 10px;
      font-size: 18px;
      font-weight: 600;
      color: #1f232a;
    }

    .subtitle {
      margin: 0 0 14px;
      font-size: 12px;
      color: var(--muted);
    }

    .form-row {
      display: grid;
      grid-template-columns: 150px 1fr;
      align-items: center;
      margin-bottom: 12px;
      min-height: 34px;
    }

    .label {
      font-size: 12px;
      color: #3d4651;
      padding-right: 12px;
      text-align: right;
      line-height: 1.4;
    }

    .field {
      width: 100%;
    }

    .input, .select, .text-box {
      width: 100%;
      min-height: 30px;
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      background: #fff;
      font: inherit;
      color: var(--text);
      padding: 0 10px;
      outline: none;
    }

    .input:focus, .select:focus, .text-box:focus {
      border-color: #8a76ff;
      box-shadow: 0 0 0 2px rgba(122, 82, 212, 0.12);
    }

    .repo {
      display: flex;
      align-items: center;
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      background: #fff;
      overflow: hidden;
    }

    .repo .icon {
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #5f6d7d;
      border-right: 1px solid var(--line);
      font-size: 12px;
      background: #fbfbfc;
    }

    .repo input {
      border: 0;
      outline: none;
      background: transparent;
      flex: 1;
      min-height: 30px;
      padding: 0 10px;
      font: inherit;
      color: var(--text);
    }

    .repo .end {
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-left: 1px solid var(--line);
      color: #778395;
      font-size: 11px;
      background: #fbfbfc;
    }

    .double {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .thin {
      font-size: 12px;
      color: #768393;
      margin-top: 3px;
      line-height: 1.5;
    }

    .region-box {
      position: relative;
      display: flex;
      align-items: center;
      min-height: 30px;
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      background: #fff;
      overflow: hidden;
    }

    .region-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: 100%;
      min-height: 30px;
      padding: 0 10px;
      font-size: 12px;
      color: #2f3b48;
    }

    .region-pill .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #a36af3;
      box-shadow: 0 0 0 2px rgba(163, 106, 243, 0.16);
      display: inline-block;
    }

    .region-box .add {
      width: 38px;
      border-left: 1px solid var(--line);
      background: #fafbfc;
      color: #667587;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      height: 100%;
    }

    .compute-card {
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      overflow: hidden;
      background: #fff;
    }

    .compute-header {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      color: #667587;
      background: #fafbfc;
    }

    .compute-header > div {
      padding: 8px 10px;
      border-right: 1px solid var(--line);
    }

    .compute-header > div:last-child {
      border-right: none;
    }

    .plan-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      font-size: 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .plan-row:last-child {
      border-bottom: none;
    }

    .plan-row.selected {
      background: var(--accent-soft);
      box-shadow: inset 0 0 0 1px #c9b5fa;
    }

    .plan-cell {
      padding: 9px 10px;
      border-right: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 34px;
      color: #2d3948;
    }

    .plan-row .plan-cell:last-child {
      border-right: none;
    }

    .radio {
      width: 12px;
      height: 12px;
      border: 1px solid #8c94a2;
      border-radius: 50%;
      display: inline-block;
      position: relative;
      background: #fff;
      flex: 0 0 auto;
    }

    .plan-row.selected .radio {
      border-color: var(--accent);
    }

    .plan-row.selected .radio::after {
      content: "";
      position: absolute;
      inset: 2px;
      border-radius: 50%;
      background: var(--accent);
    }

    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
      color: #6d4ec0;
      background: rgba(122, 82, 212, 0.08);
      border: 1px solid rgba(122, 82, 212, 0.18);
    }

    .env-box {
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      overflow: hidden;
      background: #fff;
    }

    .env-head {
      display: grid;
      grid-template-columns: 1fr 1fr 60px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      color: #6b7888;
      background: #fafbfc;
    }

    .env-head > div {
      padding: 8px 10px;
      border-right: 1px solid var(--line);
    }

    .env-head > div:last-child {
      border-right: none;
    }

    .env-row {
      display: grid;
      grid-template-columns: 1fr 1fr 60px;
      min-height: 36px;
      background: #fff;
    }

    .env-row > div {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 7px 10px;
    }

    .env-row > div:last-child {
      border-right: none;
    }

    .env-row:last-child > div {
      border-bottom: none;
    }

    .env-input {
      border: 1px solid var(--line-strong);
      border-radius: 3px;
      min-height: 26px;
      width: 100%;
      padding: 0 8px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }

    .env-action {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #6f7b8a;
      font-size: 18px;
      cursor: pointer;
      user-select: none;
    }

    .toolbar {
      margin-top: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }

    .toolbar-left {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #788495;
      font-size: 11px;
    }

    .caret {
      display: inline-block;
      width: 0;
      height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-top: 7px solid #586779;
      margin-left: 2px;
    }

    .toolbar-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn {
      min-width: 96px;
      height: 30px;
      border-radius: 4px;
      border: 1px solid var(--line-strong);
      background: #f6f7f9;
      color: #2c3948;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    .btn.primary {
      background: #1d232b;
      border-color: #1d232b;
      color: #fff;
    }

    .btn.primary:hover { background: var(--button-hover); }
    .btn:hover { background: #eff2f5; }

    .section {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }

    .section h3 {
      margin: 0 0 12px;
      font-size: 18px;
      color: #1f232b;
      font-weight: 600;
    }

    .row-compact {
      display: grid;
      grid-template-columns: 150px 1fr;
      align-items: center;
      margin-bottom: 10px;
    }

    @media (max-width: 820px) {
      .page { width: 100%; }
      .form-row, .row-compact { grid-template-columns: 1fr; }
      .label { text-align: left; padding-right: 0; margin-bottom: 6px; }
      .double { grid-template-columns: 1fr; }
      .compute-header, .compute-row, .plan-row, .env-head, .env-row { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1 class="title">New Web Service</h1>
      <p class="subtitle">It looks like you're using Python, so we automatically filled some fields accordingly.</p>

      <div class="form-row">
        <div class="label">Source Code</div>
        <div class="field">
          <div class="repo">
            <div class="icon">◌</div>
            <input type="text" value="vamshikrishna-bot/CampusCompass-AI" aria-label="Source code" />
            <div class="end">↗</div>
          </div>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Name</div>
        <div class="field">
          <input class="input" type="text" value="CampusCompass-AI" aria-label="Name"/>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Project<br><span style="color:#788395; font-size:11px;">Optional</span></div>
        <div class="field double">
          <select class="select" aria-label="Project">
            <option>Select a project...</option>
          </select>
          <select class="select" aria-label="Environment">
            <option>Select an environment...</option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Language</div>
        <div class="field">
          <select class="select" aria-label="Language">
            <option selected>Python 3.11</option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Branch</div>
        <div class="field">
          <input class="input" type="text" value="main" aria-label="Branch"/>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Region</div>
        <div class="field">
          <div class="region-box">
            <div class="region-pill"><span class="dot"></span> East US (West Europe)</div>
            <div class="add">＋</div>
          </div>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Root Directory<br><span style="color:#788395; font-size:11px;">Optional</span></div>
        <div class="field">
          <input class="input" type="text" aria-label="Root directory"/>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Build Command</div>
        <div class="field">
          <input class="input" type="text" value="pip install -r requirements-render.txt" aria-label="Build command"/>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Start Command</div>
        <div class="field">
          <input class="input" type="text" value="uvicorn app:app --host 0.0.0.0 --port $PORT" aria-label="Start command"/>
        </div>
      </div>

      <div class="form-row">
        <div class="label">Compute</div>
        <div class="field">
          <div class="compute-card">
            <div class="compute-header">
              <div>SKU</div>
              <div>vCPU</div>
              <div>RAM</div>
              <div>Price</div>
            </div>

            <div class="plan-row selected">
              <div class="plan-cell"><span class="radio"></span> <strong>50 / month</strong></div>
              <div class="plan-cell">1 vCPU</div>
              <div class="plan-cell">512 MiB RAM</div>
              <div class="plan-cell">Free</div>
            </div>

            <div class="plan-row">
              <div class="plan-cell"><span class="radio"></span> 50 / month</div>
              <div class="plan-cell">0.5 vCPU</div>
              <div class="plan-cell">512 MiB RAM</div>
              <div class="plan-cell">$ 0.5-0.5</div>
            </div>

            <div class="plan-row">
              <div class="plan-cell"><span class="radio"></span> 125 / month</div>
              <div class="plan-cell">1 vCPU</div>
              <div class="plan-cell">2 GiB RAM</div>
              <div class="plan-cell">$ 1-2</div>
            </div>

            <div class="plan-row">
              <div class="plan-cell"><span class="radio"></span> 250 / month</div>
              <div class="plan-cell">2 vCPU</div>
              <div class="plan-cell">4 GiB RAM</div>
              <div class="plan-cell">$ 2-4</div>
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <h3>Environment Variables</h3>
        <div class="env-box">
          <div class="env-head">
            <div>Key</div>
            <div>Value</div>
            <div></div>
          </div>
          <div class="env-row">
            <div><input class="env-input" type="text" value="NAME_OF_VARIABLE" aria-label="Environment key" /></div>
            <div><input class="env-input" type="text" value="value" aria-label="Environment value" /></div>
            <div class="env-action">✦</div>
          </div>
        </div>

        <div class="toolbar">
          <div class="toolbar-left"><span>+ Add environment variable</span></div>
          <div class="toolbar-right">
            <button class="btn" type="button">Add from env</button>
            <button class="btn primary" type="button">Create</button>
          </div>
        </div>
      </div>

      <div class="section" style="margin-top:8px; border-top:1px solid var(--line); padding-top: 10px;">
        <div class="toolbar" style="margin-top:0; border-top:none; padding-top:0;">
          <div class="toolbar-left"><span style="font-size:12px; font-weight:600; color:#272f39;">Advanced</span> <span class="caret"></span></div>
          <div class="toolbar-right" style="gap:6px;">
            <button class="btn" type="button">Deploy web service</button>
            <button class="btn primary" type="button" style="min-width: 74px;">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  </div>
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
