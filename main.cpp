/*
 * VectorDB — Vector Database from Scratch in C++
 * ─────────────────────────────────────────────────
 * Algorithms : BruteForce | KD-Tree | HNSW
 * Metrics    : Cosine | Euclidean | Manhattan
 * RAG        : Ollama  (nomic-embed-text + llama3.2)
 *
 * Build — Windows (MinGW-w64):
 *   g++ -std=c++17 -O2 main.cpp -o db.exe -lws2_32
 *
 * Build — Linux / Mac:
 *   g++ -std=c++17 -O2 main.cpp -o db
 *
 * Run:
 *   ./db.exe        → open http://localhost:8080
 */

//#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ════════════════════════════════════════════════════════════════
//  SECTION 1 — DISTANCE METRICS
// ════════════════════════════════════════════════════════════════

using Vec    = vector<double>;
using DistFn = function<double(const Vec&, const Vec&)>;

double dist_cosine(const Vec& a, const Vec& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    // Returns cosine DISTANCE: 0 = identical, 2 = opposite
    return 1.0 - dot / (sqrt(na) * sqrt(nb) + 1e-9);
}

double dist_euclidean(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i]; s += d * d;
    }
    return sqrt(s);
}

double dist_manhattan(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) s += fabs(a[i] - b[i]);
    return s;
}

DistFn get_metric(const string& name) {
    if (name == "euclidean") return dist_euclidean;
    if (name == "manhattan") return dist_manhattan;
    return dist_cosine; // default
}

// ════════════════════════════════════════════════════════════════
//  SECTION 2 — SHARED DATA STRUCTURES
// ════════════════════════════════════════════════════════════════

struct VecItem {
    int    id;
    string label;
    string category;
    Vec    v;
};

struct SearchResult {
    int    id;
    string label;
    string category;
    double dist;
};

// ════════════════════════════════════════════════════════════════
//  SECTION 3 — BRUTE FORCE  O(N·d)
// ════════════════════════════════════════════════════════════════

class BruteForce {
public:
    vector<VecItem> items;
    int next_id = 0;

    int insert(const string& label, const string& cat, const Vec& v) {
        items.push_back({next_id, label, cat, v});
        return next_id++;
    }

    bool remove(int id) {
        auto it = remove_if(items.begin(), items.end(),
                    [id](const VecItem& x){ return x.id == id; });
        if (it == items.end()) return false;
        items.erase(it, items.end());
        return true;
    }

    vector<SearchResult> search(const Vec& q, int k, const DistFn& fn) const {
        vector<SearchResult> res;
        for (auto& item : items)
            res.push_back({item.id, item.label, item.category, fn(q, item.v)});
        sort(res.begin(), res.end(), [](auto& a, auto& b){ return a.dist < b.dist; });
        if ((int)res.size() > k) res.resize(k);
        return res;
    }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 4 — KD-TREE  O(log N) exact
// ════════════════════════════════════════════════════════════════

class KDTree {
    struct Node {
        VecItem item;
        int left = -1, right = -1;
        int split_dim = 0;
    };
    vector<Node> nodes;
    int root = -1;
    int dims = 0;

    int build(vector<int>& idx, vector<VecItem>& pts, int depth) {
        if (idx.empty()) return -1;
        int dim = depth % dims;
        sort(idx.begin(), idx.end(),
            [&](int a, int b){ return pts[a].v[dim] < pts[b].v[dim]; });
        int mid = (int)idx.size() / 2;
        Node n; n.item = pts[idx[mid]]; n.split_dim = dim;
        int ni = (int)nodes.size(); nodes.push_back(n);
        vector<int> L(idx.begin(), idx.begin() + mid);
        vector<int> R(idx.begin() + mid + 1, idx.end());
        nodes[ni].left  = build(L, pts, depth + 1);
        nodes[ni].right = build(R, pts, depth + 1);
        return ni;
    }

    using PQ = priority_queue<pair<double,int>>;

    void knn(int node, const Vec& q, int k, PQ& pq, const DistFn& fn) const {
        if (node < 0) return;
        const Node& n = nodes[node];
        double d = fn(q, n.item.v);
        if ((int)pq.size() < k || d < pq.top().first) {
            pq.push({d, node});
            if ((int)pq.size() > k) pq.pop();
        }
        double diff = q[n.split_dim] - n.item.v[n.split_dim];
        int first  = diff < 0 ? n.left  : n.right;
        int second = diff < 0 ? n.right : n.left;
        knn(first, q, k, pq, fn);
        if ((int)pq.size() < k || fabs(diff) < pq.top().first)
            knn(second, q, k, pq, fn);
    }

public:
    BruteForce store; // raw data for rebuild

    void rebuild(int d) {
        dims = d; nodes.clear(); root = -1;
        if (store.items.empty()) return;
        vector<int> idx(store.items.size());
        iota(idx.begin(), idx.end(), 0);
        root = build(idx, store.items, 0);
    }

    int insert(const string& label, const string& cat, const Vec& v) {
        dims = (int)v.size();
        int id = store.insert(label, cat, v);
        rebuild(dims);
        return id;
    }

    bool remove(int id) {
        bool ok = store.remove(id);
        if (ok && dims > 0) rebuild(dims);
        return ok;
    }

    vector<SearchResult> search(const Vec& q, int k, const DistFn& fn) const {
        if (root < 0) return {};
        PQ pq;
        knn(root, q, k, pq, fn);
        vector<SearchResult> res;
        while (!pq.empty()) {
            int ni = pq.top().second; pq.pop();
            auto& item = nodes[ni].item;
            res.push_back({item.id, item.label, item.category, fn(q, item.v)});
        }
        sort(res.begin(), res.end(), [](auto& a, auto& b){ return a.dist < b.dist; });
        return res;
    }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 5 — HNSW  O(log N) approximate
// ════════════════════════════════════════════════════════════════

class HNSW {
    struct HNode {
        VecItem item;
        vector<vector<int>> neighbors; // [layer] → neighbor indices
    };

    vector<HNode> nodes;
    int entry     = -1;
    int max_layer = -1;

    static const int M            = 16;
    static const int M0           = 32;
    static const int EF_CONSTRUCT = 200;
    const double ml = 1.0 / log(2.0);
    mt19937 rng{42};

    int random_level() {
        uniform_real_distribution<double> dist(0.0, 1.0);
        return (int)(-log(dist(rng)) * ml);
    }

    using PQMin = priority_queue<pair<double,int>,
                    vector<pair<double,int>>, greater<pair<double,int>>>;
    using PQMax = priority_queue<pair<double,int>>;

    vector<int> search_layer(const Vec& q, int ep, int ef, int lc,
                              const DistFn& fn) const {
        vector<bool> visited(nodes.size(), false);
        PQMin candidates;
        PQMax found;
        double d0 = fn(q, nodes[ep].item.v);
        candidates.push({d0, ep}); found.push({d0, ep}); visited[ep] = true;

        while (!candidates.empty()) {
            auto [dc, c] = candidates.top(); candidates.pop();
            if (dc > found.top().first) break;
            if (lc >= (int)nodes[c].neighbors.size()) continue;
            for (int nb : nodes[c].neighbors[lc]) {
                if (visited[nb]) continue;
                visited[nb] = true;
                double dn = fn(q, nodes[nb].item.v);
                if ((int)found.size() < ef || dn < found.top().first) {
                    candidates.push({dn, nb}); found.push({dn, nb});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }
        vector<int> res;
        while (!found.empty()) { res.push_back(found.top().second); found.pop(); }
        sort(res.begin(), res.end(), [&](int a, int b){
            return fn(q, nodes[a].item.v) < fn(q, nodes[b].item.v);
        });
        return res;
    }

    void connect(int idx, const vector<int>& cands, int lc, const DistFn& fn) {
        int Mmax = lc == 0 ? M0 : M;
        vector<int> sel(cands.begin(),
            cands.begin() + min((int)cands.size(), Mmax));

        if (lc >= (int)nodes[idx].neighbors.size())
            nodes[idx].neighbors.resize(lc + 1);
        nodes[idx].neighbors[lc] = sel;

        for (int nb : sel) {
            if (lc >= (int)nodes[nb].neighbors.size())
                nodes[nb].neighbors.resize(lc + 1);
            nodes[nb].neighbors[lc].push_back(idx);
            if ((int)nodes[nb].neighbors[lc].size() > Mmax) {
                auto& nbs = nodes[nb].neighbors[lc];
                sort(nbs.begin(), nbs.end(), [&](int a, int b){
                    return fn(nodes[nb].item.v, nodes[a].item.v) <
                           fn(nodes[nb].item.v, nodes[b].item.v);
                });
                nbs.resize(Mmax);
            }
        }
    }

public:
    int insert(const string& label, const string& cat,
               const Vec& v, const DistFn& fn) {
        static int gid = 0;
        int id = gid++;
        int idx = (int)nodes.size();
        HNode hn; hn.item = {id, label, cat, v};
        int lc = random_level();
        hn.neighbors.resize(lc + 1);
        nodes.push_back(hn);

        if (entry < 0) { entry = 0; max_layer = lc; return id; }

        int ep = entry;
        for (int l = max_layer; l > lc; --l) {
            auto res = search_layer(v, ep, 1, l, fn);
            if (!res.empty()) ep = res[0];
        }
        for (int l = min(lc, max_layer); l >= 0; --l) {
            auto cands = search_layer(v, ep, EF_CONSTRUCT, l, fn);
            connect(idx, cands, l, fn);
            if (!cands.empty()) ep = cands[0];
        }
        if (lc > max_layer) { entry = idx; max_layer = lc; }
        return id;
    }

    vector<SearchResult> search(const Vec& q, int k, const DistFn& fn) const {
        if (entry < 0) return {};
        int ep = entry;
        for (int l = max_layer; l > 0; --l) {
            auto res = search_layer(q, ep, 1, l, fn);
            if (!res.empty()) ep = res[0];
        }
        auto cands = search_layer(q, ep, max(k, EF_CONSTRUCT), 0, fn);
        vector<SearchResult> res;
        for (int i = 0; i < min(k, (int)cands.size()); i++) {
            auto& item = nodes[cands[i]].item;
            res.push_back({item.id, item.label, item.category, fn(q, item.v)});
        }
        return res;
    }

    string info_json() const {
        map<int,int> lcount;
        for (auto& n : nodes) lcount[(int)n.neighbors.size() - 1]++;
        ostringstream oss;
        oss << "{\"total_nodes\":" << nodes.size()
            << ",\"max_layer\":"   << max_layer
            << ",\"M\":"           << M << ",\"M0\":" << M0
            << ",\"ef_construction\":" << EF_CONSTRUCT
            << ",\"layers\":[";
        bool first = true;
        for (auto& [l, c] : lcount) {
            if (!first) oss << ",";
            oss << "{\"layer\":" << l << ",\"nodes\":" << c << "}";
            first = false;
        }
        oss << "]}"; return oss.str();
    }

    int size() const { return (int)nodes.size(); }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 6 — DEMO DATABASE  (16-D, 20 vectors)
// ════════════════════════════════════════════════════════════════

struct BenchResult {
    vector<SearchResult> hnsw_res, kd_res, bf_res;
    double hnsw_us = 0, kd_us = 0, bf_us = 0;
};

class DemoDB {
public:
    mutex      mtx;
    BruteForce bf;
    KDTree     kd;
    HNSW       hnsw;

    void insert(const string& label, const string& cat, const Vec& v) {
        bf.insert(label, cat, v);
        kd.insert(label, cat, v);
        hnsw.insert(label, cat, v, dist_cosine);
    }

    vector<SearchResult> search(const Vec& q, int k,
                                 const string& metric, const string& algo) {
        lock_guard<mutex> lk(mtx);
        DistFn fn = get_metric(metric);
        if (algo == "kdtree")     return kd.search(q, k, fn);
        if (algo == "bruteforce") return bf.search(q, k, fn);
        return hnsw.search(q, k, fn);
    }

    BenchResult benchmark(const Vec& q, int k, const string& metric) {
        lock_guard<mutex> lk(mtx);
        DistFn fn = get_metric(metric);
        BenchResult br;
        auto now = []{ return chrono::high_resolution_clock::now(); };
        auto us  = [](auto a, auto b){
            return chrono::duration<double, micro>(b - a).count();
        };
        auto t0 = now(); br.hnsw_res = hnsw.search(q, k, fn);
        auto t1 = now(); br.kd_res   = kd.search(q, k, fn);
        auto t2 = now(); br.bf_res   = bf.search(q, k, fn);
        auto t3 = now();
        br.hnsw_us = us(t0,t1); br.kd_us = us(t1,t2); br.bf_us = us(t2,t3);
        return br;
    }

    string stats_json() {
        lock_guard<mutex> lk(mtx);
        return "{\"count\":" + to_string(bf.items.size()) + ",\"dims\":16}";
    }

    string list_json() {
        lock_guard<mutex> lk(mtx);
        ostringstream oss; oss << "[";
        for (size_t i = 0; i < bf.items.size(); i++) {
            if (i) oss << ",";
            auto& it = bf.items[i];
            oss << "{\"id\":"         << it.id
                << ",\"label\":\""    << it.label    << "\""
                << ",\"category\":\"" << it.category << "\""
                << ",\"dims\":"       << it.v.size() << "}";
        }
        oss << "]"; return oss.str();
    }

    string hnsw_info() { return hnsw.info_json(); }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 7 — JSON / STRING HELPERS
// ════════════════════════════════════════════════════════════════

static string json_escape(const string& s) {
    string out;
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

// Parse first JSON number array [...] (handles nested brackets)
static Vec parse_number_array(const string& s, const string& key = "") {
    size_t pos = string::npos;
    if (!key.empty()) {
        size_t kp = s.find("\"" + key + "\"");
        if (kp != string::npos) pos = s.find('[', kp);
    }
    if (pos == string::npos) pos = s.find('[');
    if (pos == string::npos) return {};

    int depth = 0; size_t end = pos;
    for (; end < s.size(); end++) {
        if (s[end]=='[') depth++;
        else if (s[end]==']') { depth--; if (!depth) break; }
    }
    string arr = s.substr(pos + 1, end - pos - 1);
    stringstream ss(arr); string tok; Vec vec;
    while (getline(ss, tok, ','))
        try { vec.push_back(stod(tok)); } catch(...) {}
    return vec;
}

// Extract "response" field from Ollama non-streaming reply
static string parse_response_field(const string& s) {
    size_t p = s.find("\"response\"");
    if (p == string::npos) return "No response field.";
    size_t q1 = s.find('"', p + 11);
    if (q1 == string::npos) return "Parse error.";
    string out;
    for (size_t i = q1 + 1; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char nc = s[++i];
            if      (nc == 'n') out += '\n';
            else if (nc == 't') out += '\t';
            else if (nc == 'r') out += '\r';
            else                out += nc;
        } else if (s[i] == '"') break;
        else out += s[i];
    }
    return out;
}

// Parse JSON string field: "key":"value"  (handles basic escapes)
static string json_str_field(const string& body, const string& key) {
    string pat = "\"" + key + "\"";
    size_t p = body.find(pat);
    if (p == string::npos) return "";
    size_t colon = body.find(':', p + pat.size());
    size_t q1    = body.find('"', colon + 1);
    if (q1 == string::npos) return "";
    string out;
    for (size_t i = q1 + 1; i < body.size(); i++) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            char nc = body[++i];
            if      (nc == 'n') out += '\n';
            else if (nc == 't') out += '\t';
            else                out += nc;
        } else if (body[i] == '"') break;
        else out += body[i];
    }
    return out;
}

static string qparam(const httplib::Request& req, const string& key,
                     const string& def = "") {
    return req.has_param(key) ? req.get_param_value(key) : def;
}

static Vec parse_vec_str(const string& s) {
    Vec v; stringstream ss(s); string tok;
    while (getline(ss, tok, ','))
        try { v.push_back(stod(tok)); } catch(...) {}
    return v;
}

// ════════════════════════════════════════════════════════════════
//  SECTION 8 — OLLAMA CLIENT  (fixed: host/port, stream:false, parse)
// ════════════════════════════════════════════════════════════════

struct OllamaClient {
    const string host        = "127.0.0.1"; // FIX: not "http://localhost:11434"
    const int    port        = 11434;
    const string embed_model = "nomic-embed-text";
    string       gen_model   = "llama3.2"; // change to "llama3.2:1b" for speed

    bool is_online() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    Vec embed(const string& text) {
        httplib::Client cli(host, port);
        cli.set_read_timeout(60, 0);
        // FIX: json_escape prevents JSON breakage on quotes/newlines in text
        string body = "{\"model\":\"" + embed_model + "\","
                      "\"prompt\":\"" + json_escape(text) + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) return {};
        return parse_number_array(res->body, "embedding");
    }

    string generate(const string& context, const string& question) {
        httplib::Client cli(host, port);
        cli.set_read_timeout(120, 0);
        string prompt = "Answer using ONLY this context:\\n\\n" +
                        json_escape(context) +
                        "\\n\\nQuestion: " + json_escape(question);
        // FIX: stream:false so we get one JSON response, not streaming lines
        string body = "{\"model\":\""  + gen_model + "\","
                      "\"prompt\":\""  + prompt    + "\","
                      "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200)
            return "LLM error (HTTP " +
                   (res ? to_string(res->status) : "no conn") + ")";
        return parse_response_field(res->body);
    }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 9 — DOCUMENT DATABASE  (768-D real Ollama embeddings)
// ════════════════════════════════════════════════════════════════

struct DocChunk {
    int    id;
    string title;
    string text;
    Vec    embedding;
};

class DocumentDB {
public:
    mutex         mtx;
    vector<DocChunk> chunks;
    HNSW          hnsw_index;
    int           next_id = 0;
    OllamaClient  ollama;

    // Split into ~250-word overlapping chunks
    static vector<string> chunk_text(const string& text,
                                      int size = 250, int overlap = 50) {
        istringstream iss(text);
        vector<string> words;
        string w;
        while (iss >> w) words.push_back(w);

        vector<string> result;
        for (int i = 0; i < (int)words.size(); i += size - overlap) {
            string chunk;
            int end = min((int)words.size(), i + size);
            for (int j = i; j < end; j++) {
                if (j > i) chunk += ' ';
                chunk += words[j];
            }
            result.push_back(chunk);
            if (end >= (int)words.size()) break;
        }
        return result;
    }

    // Returns chunks inserted, -1 on embed failure
    int insert_doc(const string& title, const string& text) {
        auto parts = chunk_text(text);
        int inserted = 0;
        for (auto& part : parts) {
            Vec emb = ollama.embed(part);
            if (emb.empty()) return -1;
            lock_guard<mutex> lk(mtx);
            hnsw_index.insert(title, to_string(next_id), emb, dist_cosine);
            chunks.push_back({next_id++, title, part, emb});
            inserted++;
        }
        return inserted;
    }

    pair<string, vector<DocChunk>> ask(const string& question, int k = 3) {
        Vec q_emb = ollama.embed(question);
        if (q_emb.empty()) return {"Embedding failed — is Ollama running?", {}};

        vector<SearchResult> hits;
        {
            lock_guard<mutex> lk(mtx);
            hits = hnsw_index.search(q_emb, k, dist_cosine);
        }

        string context;
        vector<DocChunk> used;
        for (auto& h : hits) {
            lock_guard<mutex> lk(mtx);
            for (auto& c : chunks) {
                if (c.title == h.label && context.find(c.text) == string::npos) {
                    context += c.text + "\n---\n";
                    used.push_back(c);
                    break;
                }
            }
        }
        string answer = ollama.generate(context, question);
        return {answer, used};
    }

    bool remove_doc(int id) {
        lock_guard<mutex> lk(mtx);
        auto it = remove_if(chunks.begin(), chunks.end(),
                    [id](const DocChunk& c){ return c.id == id; });
        if (it == chunks.end()) return false;
        chunks.erase(it, chunks.end());
        return true;
    }

    string list_json() {
        lock_guard<mutex> lk(mtx);
        ostringstream oss; oss << "[";
        for (size_t i = 0; i < chunks.size(); i++) {
            if (i) oss << ",";
            auto& c = chunks[i];
            oss << "{\"id\":"       << c.id
                << ",\"title\":\"" << json_escape(c.title) << "\""
                << ",\"chars\":"   << c.text.size()
                << ",\"dims\":"    << c.embedding.size() << "}";
        }
        oss << "]"; return oss.str();
    }
};

// ════════════════════════════════════════════════════════════════
//  SECTION 10 — HTTP RESPONSE HELPERS
// ════════════════════════════════════════════════════════════════

static void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}
static void json_ok(httplib::Response& res, const string& body) {
    cors(res); res.status = 200;
    res.set_content(body, "application/json");
}
static void json_err(httplib::Response& res, int code, const string& msg) {
    cors(res); res.status = code;
    res.set_content("{\"error\":\"" + json_escape(msg) + "\"}", "application/json");
}

static string results_json(const vector<SearchResult>& r) {
    ostringstream o; o << "[";
    for (size_t i = 0; i < r.size(); i++) {
        if (i) o << ",";
        o << "{\"id\":"         << r[i].id
          << ",\"label\":\""    << json_escape(r[i].label)    << "\""
          << ",\"category\":\"" << json_escape(r[i].category) << "\""
          << ",\"dist\":"       << r[i].dist << "}";
    }
    o << "]"; return o.str();
}

static string bench_json(const BenchResult& br) {
    ostringstream o;
    o << "{\"hnsw\":{\"results\":"      << results_json(br.hnsw_res) << ",\"us\":" << br.hnsw_us << "}"
      << ",\"kdtree\":{\"results\":"    << results_json(br.kd_res)   << ",\"us\":" << br.kd_us   << "}"
      << ",\"bruteforce\":{\"results\":" << results_json(br.bf_res)  << ",\"us\":" << br.bf_us   << "}}";
    return o.str();
}

// ════════════════════════════════════════════════════════════════
//  SECTION 11 — SEED DEMO VECTORS  (16-D, 4 categories, 20 items)
// ════════════════════════════════════════════════════════════════

static void seed_demo(DemoDB& db) {
    // Dims 0-3 = CS | 4-7 = Math | 8-11 = Food | 12-15 = Sports
    struct S { string label, cat; Vec v; };
    vector<S> seeds = {
      // Computer Science
      {"binary tree",   "CS",     {.9,.8,.7,.6, .1,.1,.1,.1, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"linked list",   "CS",     {.8,.9,.6,.7, .1,.2,.1,.1, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"hash table",    "CS",     {.7,.6,.9,.8, .2,.1,.1,.1, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"neural net",    "CS",     {.8,.7,.8,.9, .1,.1,.1,.2, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"quicksort",     "CS",     {.6,.7,.8,.7, .1,.1,.2,.1, .1,.1,.1,.1, .1,.1,.1,.1}},
      // Mathematics
      {"calculus",      "Math",   {.1,.1,.1,.1, .9,.8,.7,.6, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"linear alg",    "Math",   {.1,.2,.1,.1, .8,.9,.6,.7, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"probability",   "Math",   {.1,.1,.2,.1, .7,.6,.9,.8, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"topology",      "Math",   {.2,.1,.1,.1, .6,.7,.8,.9, .1,.1,.1,.1, .1,.1,.1,.1}},
      {"number theory", "Math",   {.1,.1,.1,.2, .8,.8,.7,.7, .1,.1,.1,.1, .1,.1,.1,.1}},
      // Food
      {"sushi",         "Food",   {.1,.1,.1,.1, .1,.1,.1,.1, .9,.8,.7,.6, .1,.1,.1,.1}},
      {"pizza",         "Food",   {.1,.1,.1,.1, .1,.2,.1,.1, .8,.9,.6,.7, .1,.1,.1,.1}},
      {"ramen",         "Food",   {.1,.1,.1,.1, .1,.1,.2,.1, .7,.6,.9,.8, .1,.1,.1,.1}},
      {"biryani",       "Food",   {.1,.1,.1,.1, .1,.1,.1,.2, .6,.7,.8,.9, .1,.1,.1,.1}},
      {"tacos",         "Food",   {.1,.1,.1,.1, .2,.1,.1,.1, .8,.7,.7,.8, .1,.1,.1,.1}},
      // Sports
      {"basketball",    "Sports", {.1,.1,.1,.1, .1,.1,.1,.1, .1,.1,.1,.1, .9,.8,.7,.6}},
      {"cricket",       "Sports", {.1,.1,.1,.1, .1,.2,.1,.1, .1,.1,.1,.1, .8,.9,.6,.7}},
      {"football",      "Sports", {.1,.1,.1,.1, .1,.1,.2,.1, .1,.1,.1,.1, .7,.6,.9,.8}},
      {"tennis",        "Sports", {.1,.1,.1,.1, .1,.1,.1,.2, .1,.1,.1,.1, .6,.7,.8,.9}},
      {"swimming",      "Sports", {.1,.1,.1,.1, .2,.1,.1,.1, .1,.1,.1,.1, .8,.7,.7,.8}},
    };
    for (auto& s : seeds) db.insert(s.label, s.cat, s.v);
    cout << "  Seeded " << seeds.size() << " demo vectors — 16D, 4 categories\n";
}

// ════════════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════════════

int main() {
    cout << "\n"
         << "╔══════════════════════════════════════════════╗\n"
         << "║      VectorDB — from scratch in C++          ║\n"
         << "║  HNSW | KD-Tree | BruteForce | RAG (Ollama) ║\n"
         << "╚══════════════════════════════════════════════╝\n\n";

    static DemoDB     demoDB;
    static DocumentDB docDB;
    static OllamaClient ollama;

    cout << "  Loading demo vectors...\n";
    seed_demo(demoDB);

    bool ollamaOnline = ollama.is_online();
    cout << "  Ollama: " << (ollamaOnline ? "ONLINE ✓" : "OFFLINE ✗  (RAG disabled)") << "\n";
    if (ollamaOnline)
        cout << "    embed: nomic-embed-text\n"
             << "    gen  : llama3.2\n";

    httplib::Server svr;
    svr.set_base_dir("."); // serve index.html from current folder

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res){
        cors(res); res.status = 204;
    });

    // ── /status ───────────────────────────────────────────────
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res){
        bool on = ollama.is_online();
        json_ok(res, "{\"ollama\":"    + string(on?"true":"false") +
                     ",\"embed_model\":\"nomic-embed-text\""
                     ",\"gen_model\":\"llama3.2\"}");
    });

    // ── /stats ────────────────────────────────────────────────
    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res){
        json_ok(res, demoDB.stats_json());
    });

    // ── /items ────────────────────────────────────────────────
    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res){
        json_ok(res, demoDB.list_json());
    });

    // ── /hnsw-info ────────────────────────────────────────────
    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res){
        json_ok(res, demoDB.hnsw_info());
    });

    // ── GET /search?v=...&k=5&metric=cosine&algo=hnsw ─────────
    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res){
        string vs     = qparam(req, "v");
        string metric = qparam(req, "metric", "cosine");
        string algo   = qparam(req, "algo",   "hnsw");
        int k = stoi(qparam(req, "k", "5"));
        if (vs.empty()) return json_err(res, 400, "missing ?v=");
        Vec q = parse_vec_str(vs);
        if (q.empty()) return json_err(res, 400, "could not parse vector");
        json_ok(res, results_json(demoDB.search(q, k, metric, algo)));
    });

    // ── GET /benchmark?v=...&k=5&metric=cosine ────────────────
    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res){
        string vs     = qparam(req, "v");
        string metric = qparam(req, "metric", "cosine");
        int k = stoi(qparam(req, "k", "5"));
        if (vs.empty()) return json_err(res, 400, "missing ?v=");
        Vec q = parse_vec_str(vs);
        if (q.empty()) return json_err(res, 400, "could not parse vector");
        json_ok(res, bench_json(demoDB.benchmark(q, k, metric)));
    });

    // ── POST /insert  {"label":"x","category":"y","vector":[...]} ──
    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res){
        string label = json_str_field(req.body, "label");
        string cat   = json_str_field(req.body, "category");
        Vec v = parse_number_array(req.body, "vector");
        if (v.empty()) return json_err(res, 400, "empty or missing vector");
        demoDB.insert(label, cat, v);
        json_ok(res, "{\"status\":\"inserted\",\"dims\":" + to_string(v.size()) + "}");
    });

    // ── DELETE /delete/:id ────────────────────────────────────
    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res){
        int id = stoi(req.matches[1]);
        demoDB.bf.remove(id) ?
            json_ok(res, "{\"status\":\"deleted\",\"id\":" + to_string(id) + "}") :
            json_err(res, 404, "id not found");
    });

    // ── POST /doc/insert  {"title":"...","text":"..."} ────────
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res){
        string title = json_str_field(req.body, "title");
        string text  = json_str_field(req.body, "text");
        if (text.empty()) return json_err(res, 400, "text field missing");
        int n = docDB.insert_doc(title.empty() ? "Untitled" : title, text);
        if (n < 0) return json_err(res, 503,
            "Ollama embedding failed — run: ollama serve");
        json_ok(res, "{\"status\":\"inserted\",\"chunks\":" + to_string(n) + "}");
    });

    // ── GET /doc/list ─────────────────────────────────────────
    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res){
        json_ok(res, docDB.list_json());
    });

    // ── DELETE /doc/delete/:id ────────────────────────────────
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res){
        int id = stoi(req.matches[1]);
        docDB.remove_doc(id) ?
            json_ok(res, "{\"status\":\"deleted\"}") :
            json_err(res, 404, "id not found");
    });

    // ── POST /doc/ask  {"question":"...","k":3} ───────────────
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res){
        string question = json_str_field(req.body, "question");
        if (question.empty()) return json_err(res, 400, "question field missing");
        int k = 3;
        size_t kp = req.body.find("\"k\"");
        if (kp != string::npos) {
            size_t cp = req.body.find(':', kp);
            try { k = stoi(req.body.substr(cp + 1)); } catch(...) {}
        }
        auto [answer, used_chunks] = docDB.ask(question, k);
        ostringstream oss;
        oss << "{\"answer\":\"" << json_escape(answer) << "\",\"context\":[";
        for (size_t i = 0; i < used_chunks.size(); i++) {
            if (i) oss << ",";
            auto& c = used_chunks[i];
            string preview = c.text.size() > 200 ? c.text.substr(0,200) + "..." : c.text;
            oss << "{\"id\":"      << c.id
                << ",\"title\":\"" << json_escape(c.title)   << "\""
                << ",\"text\":\""  << json_escape(preview)   << "\"}";
        }
        oss << "]}";
        json_ok(res, oss.str());
    });

    // ─────────────────────────────────────────────────────────
    cout << "\n  ✓ Server ready → http://localhost:8080\n\n";
    svr.listen("0.0.0.0", 8080);
    return 0;
} 