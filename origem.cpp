#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

extern "C" {
#include "vc.h"
}

void vc_timer(void) {
    static bool running = false;
    static std::chrono::steady_clock::time_point previousTime = std::chrono::steady_clock::now();

    if (!running) {
        running = true;
        previousTime = std::chrono::steady_clock::now();
    } else {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = currentTime - previousTime;
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(elapsedTime);
        std::cout << "Tempo decorrido: " << time_span.count() << " segundos\n";
        std::cout << "Pressione ENTER para terminar...\n";
        std::cin.get();
    }
}

struct VideoInfo {
    int width, height;
    int ntotalframes;
    int fps;
    int nframe;
};

struct OrangeTrack {
    cv::Point position;
    int diameter_mm;
    int ttl;
    int max_ttl;
    int hits;
    bool counted;
    bool matched;
};

struct CountedOrange {
    cv::Point position;
    int diameter_mm;
    int last_frame;
};

typedef struct {
    float dmin_mm;
    float dmax_mm;
    const char* name;
} CalibreRange;

static const CalibreRange ORANGE_CALIBRES[] = {
    {100.0f, 999.0f, "0"},
    {87.0f, 100.0f, "1"},
    {84.0f, 96.0f, "2"},
    {81.0f, 92.0f, "3"},
    {77.0f, 88.0f, "4"},
    {73.0f, 84.0f, "5"},
    {70.0f, 80.0f, "6"},
    {67.0f, 76.0f, "7"},
    {64.0f, 73.0f, "8"},
    {62.0f, 70.0f, "9"},
    {60.0f, 68.0f, "10"},
    {58.0f, 66.0f, "11"},
    {56.0f, 63.0f, "12"},
    {53.0f, 60.0f, "13"}
};

static inline void draw_text(cv::Mat& frame, const std::string& text, int x, int y,
    double scale = 0.6, int bg_thick = 2, int fg_thick = 1)
{
    cv::Point p(x, y);
    cv::putText(frame, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), bg_thick);
    cv::putText(frame, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), fg_thick);
}

static inline float px_to_mm(float px) { return px * (55.0f / 280.0f); }
static inline float diameter_px_from_area(int area_px) { return 2.0f * sqrtf((float)area_px / (float)M_PI); }
static inline float diameter_mm_from_area(int area_px) { return px_to_mm(diameter_px_from_area(area_px)); }

static inline float circularity_from_blob(const OVC& b)
{
    if (b.perimetro <= 0) return 0.0f;
    return (4.0f * (float)M_PI * (float)b.area) / ((float)b.perimetro * (float)b.perimetro);
}

static inline float deformation_percent_from_blob(const OVC& b)
{
    float w = (float)MAX(b.width, 1);
    float h = (float)MAX(b.height, 1);
    float diff = fabsf(w - h);
    return (diff / MAX(w, h)) * 100.0f;
}

static inline const char* category_from_deformation(float d)
{
    if (d <= 0.35f) return "EXTRA";
    if (d <= 0.65f) return "I";
    if (d <= 1.00f) return "II";
    return "III";
}

static inline const char* category_approved_text(const char* cat)
{
    return (cat && strcmp(cat, "III") != 0) ? "SIM" : "NAO";
}

static inline const char* calibre_from_diameter_mm(float dmm)
{
    if (dmm < 53.0f) return "AbaixoMin";
    for (int i = 0; i < (int)(sizeof(ORANGE_CALIBRES) / sizeof(ORANGE_CALIBRES[0])); i++) {
        if (dmm >= ORANGE_CALIBRES[i].dmin_mm && dmm <= ORANGE_CALIBRES[i].dmax_mm)
            return ORANGE_CALIBRES[i].name;
    }
    return "N/A";
}

static inline const char* calibre_approved_text(float dmm)
{
    if (dmm >= 53.0f) return "SIM";
    if (dmm >= 50.0f) return "NA TOLERANCIA";
    return "NAO";
}

static std::string format_float(float value, int decimals = 2)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << value;
    return oss.str();
}

static inline float point_distance(const cv::Point& a, const cv::Point& b)
{
    float dx = (float)(a.x - b.x);
    float dy = (float)(a.y - b.y);
    return sqrtf(dx * dx + dy * dy);
}

static int find_similar_counted_memory(const std::vector<CountedOrange>& memory,
    const cv::Point& position, int diameter_mm, int current_frame)
{
    int best = -1;
    float best_score = 1e30f;

    for (int i = 0; i < (int)memory.size(); i++) {
        int age = current_frame - memory[i].last_frame;
        if (age < 0 || age > 320) continue;

        int dd = abs(diameter_mm - memory[i].diameter_mm);
        if (dd > 10) continue;

        float dist = point_distance(position, memory[i].position);
        if (dist > 260.0f) continue;

        float score = dist + 14.0f * (float)dd + 0.15f * (float)age;
        if (score < best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

static bool is_orange_blob_rgb(const IVC* image_rgb, const IVC* labels, const OVC& blob)
{
    if (!image_rgb || !labels || !image_rgb->data || !labels->data) return false;
    if (image_rgb->channels != 3 || labels->channels != 1) return false;

    const unsigned char* rgb = image_rgb->data;
    const unsigned char* lbl = labels->data;
    const int rbpl = image_rgb->bytesperline;
    const int lbpl = labels->bytesperline;

    const int x0 = MAX(blob.x, 0);
    const int y0 = MAX(blob.y, 0);
    const int x1 = MIN(blob.x + blob.width, image_rgb->width);
    const int y1 = MIN(blob.y + blob.height, image_rgb->height);
    const unsigned char target = (unsigned char)blob.label;

    long long sumR = 0, sumG = 0, sumB = 0;
    int npixels = 0;
    int validHSV = 0;
    int orangeCoreHSV = 0;
    int orangeBroadHSV = 0;
    int redHSV = 0;
    int yellowHSV = 0;
    int greenHSV = 0;
    float sumSvalid = 0.0f;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            const long pL = y * lbpl + x;
            if (lbl[pL] != target) continue;

            const long pR = y * rbpl + x * 3;
            const float r = (float)rgb[pR + 0];
            const float g = (float)rgb[pR + 1];
            const float b = (float)rgb[pR + 2];

            sumR += (long long)r;
            sumG += (long long)g;
            sumB += (long long)b;
            npixels++;

            const float maxv = MAX3(r, g, b);
            const float minv = MIN3(r, g, b);
            float h = 0.0f;
            float s = (maxv > 0.0f) ? ((maxv - minv) * 100.0f / maxv) : 0.0f;

            if (maxv != minv) {
                if (r == maxv && g >= b) h = 60.0f * (g - b) / (maxv - minv);
                else if (r == maxv && b > g) h = 360.0f + 60.0f * (g - b) / (maxv - minv);
                else if (g == maxv) h = 120.0f + 60.0f * (b - r) / (maxv - minv);
                else h = 240.0f + 60.0f * (r - g) / (maxv - minv);
            }
            const float v = (maxv / 255.0f) * 100.0f;

            if (s < 28.0f || v < 20.0f) continue;
            validHSV++;
            sumSvalid += s;

            if (h >= 12.0f && h <= 30.0f && s >= 45.0f) orangeCoreHSV++;
            if (h >= 8.0f && h <= 35.0f) orangeBroadHSV++;
            if (h <= 8.0f || h >= 345.0f) redHSV++;
            if (h > 35.0f && h <= 60.0f) yellowHSV++;
            if (h >= 45.0f && h <= 120.0f) greenHSV++;
        }
    }

    if (npixels < 100 || validHSV < 60) return false;

    const float meanR = (float)sumR / (float)npixels;
    const float meanG = (float)sumG / (float)npixels;
    const float meanB = (float)sumB / (float)npixels;

    const float meanS = sumSvalid / (float)validHSV;
    const float orangeCoreRatio = (float)orangeCoreHSV / (float)validHSV;
    const float orangeBroadRatio = (float)orangeBroadHSV / (float)validHSV;
    const float redRatio = (float)redHSV / (float)validHSV;
    const float yellowRatio = (float)yellowHSV / (float)validHSV;
    const float greenRatio = (float)greenHSV / (float)validHSV;

    if (orangeBroadRatio < 0.52f) return false;
    if (orangeCoreRatio < 0.28f) return false;
    if (redRatio > 0.26f) return false;
    if (yellowRatio > 0.32f) return false;
    if (greenRatio > 0.14f) return false;
    if (meanS < 45.0f) return false;

    if (meanR < 100.0f || meanG < 55.0f || meanB > 132.0f) return false;
    if (meanR <= meanG * 1.04f) return false;
    if (meanG <= meanB * 1.05f) return false;
    if ((meanR - meanG) > 115.0f) return false;

    return true;
}

static void draw_scale_bar_50mm(cv::Mat& frame, int x, int y)
{
    const int length_px = (int)(50.0f * 280.0f / 55.0f + 0.5f);
    cv::line(frame, cv::Point(x, y), cv::Point(x + length_px, y), cv::Scalar(0, 0, 0), 3);
    cv::line(frame, cv::Point(x, y - 12), cv::Point(x, y + 12), cv::Scalar(0, 0, 0), 3);
    cv::line(frame, cv::Point(x + length_px, y - 12), cv::Point(x + length_px, y + 12), cv::Scalar(0, 0, 0), 3);
    draw_text(frame, "50", x + length_px / 2 - 14, y - 16, 0.9);
}

static std::string homogeneity_text_from_diameters_mm(const std::vector<int>& diams_mm)
{
    if (diams_mm.empty()) return "HOMOGENEIDADE DE CALIBRE: N/A";

    int dmin = diams_mm[0], dmax = diams_mm[0];
    for (int d : diams_mm) {
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
    }

    int diff = dmax - dmin;
    std::string level = (diff <= 7) ? "ELEVADA" : ((diff <= 11) ? "MODERADA" : "INEXISTENTE");
    return "HOMOGENEIDADE DE CALIBRE " + level + ", DIFERENCA DE TAMANHO MAXIMO: " + std::to_string(diff) + " MM";
}

void display(cv::Mat& frame, const VideoInfo& video, int total_laranjas, int laranjas_frame,
    const std::vector<int>& diameters_frame_mm)
{
    draw_text(frame, "FORAM ENCONTRADAS " + std::to_string(total_laranjas) + " LARANJAS!!!", 20, 30, 0.7);
    draw_text(frame, "LARANJAS NA FRAME: " + std::to_string(laranjas_frame), 20, 58, 0.7);

    draw_text(frame, "RESOLUCAO: " + std::to_string(video.width) + "x" + std::to_string(video.height), 20, 100, 0.65);
    draw_text(frame, "TOTAL FRAMES: " + std::to_string(video.ntotalframes) + "  FPS: " + std::to_string(video.fps), 20, 125, 0.65);
    draw_text(frame, "FRAME: " + std::to_string(video.nframe), 20, 150, 0.65);

    draw_scale_bar_50mm(frame, 55, video.height - 80);
    draw_text(frame, homogeneity_text_from_diameters_mm(diameters_frame_mm), 20, video.height - 20, 0.55);
}

int main(void)
{
    char videofile[] = "video.avi";
    cv::VideoCapture capture;
    capture.open(videofile);

    if (!capture.isOpened()) {
        std::cerr << "Erro ao abrir o ficheiro de vídeo!\n";
        return 1;
    }

    VideoInfo video{};
    video.ntotalframes = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    video.fps = (int)capture.get(cv::CAP_PROP_FPS);
    video.width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    video.height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    video.nframe = 0;

    cv::namedWindow("VC - VIDEO", cv::WINDOW_AUTOSIZE);
    vc_timer();

    const float DISTANCE_THRESHOLD = 85.0f;
    const int MAX_TTL = 25;
    const int MIN_HITS_TO_COUNT = 3;

    std::vector<OrangeTrack> tracks;
    std::vector<CountedOrange> counted_memory;
    int total_laranjas = 0;

    IVC* image = vc_image_new(video.width, video.height, 3, 255);
    IVC* bin = vc_image_new(video.width, video.height, 1, 255);
    IVC* labels = vc_image_new(video.width, video.height, 1, 255);
    if (!image || !bin || !labels) {
        std::cerr << "Erro a alocar imagens IVC!\n";
        vc_image_free(image); vc_image_free(bin); vc_image_free(labels);
        return 1;
    }

    cv::Mat frame;
    int key = 0;

    while (key != 'q') {
        capture.read(frame);
        if (frame.empty()) break;
        video.nframe++;

        memcpy(image->data, frame.data, video.width * video.height * 3);
        vc_bgr_to_rgb(image);

        vc_hsv_segmentation(image, bin, 8, 35, 35, 100, 18, 100);
        vc_binary_dilate_tunado(bin, bin, 5, 16);

        int nlabels = 0;
        OVC* blobs = vc_binary_blob_labelling(bin, labels, &nlabels);

        int laranjas_frame = 0;
        std::vector<int> diameters_frame_mm;

        for (auto& t : tracks) {
            t.ttl--;
            t.matched = false;
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
            [](const OrangeTrack& t) { return t.ttl <= 0; }), tracks.end());

        if (blobs != NULL && nlabels > 0) {
            vc_binary_blob_info(labels, blobs, nlabels);

            for (int i = 0; i < nlabels; i++) {
                if (blobs[i].area < 1800) continue;

                float circ = circularity_from_blob(blobs[i]);
                if (circ < 0.60f) continue;
                if (!is_orange_blob_rgb(image, labels, blobs[i])) continue;

                float dmm = diameter_mm_from_area(blobs[i].area);
                int dmm_round = (int)(dmm + 0.5f);
                float deform = deformation_percent_from_blob(blobs[i]);
                const char* categoria = category_from_deformation(deform);
                const char* categoria_ok = category_approved_text(categoria);
                const char* calibre = calibre_from_diameter_mm(dmm);
                const char* calibre_ok = calibre_approved_text(dmm);

                laranjas_frame++;
                diameters_frame_mm.push_back(dmm_round);

                cv::Point atual(blobs[i].xc, blobs[i].yc);
                bool found = false;
                int best_idx = -1;
                float best_dist = 1e30f;

                for (int ti = 0; ti < (int)tracks.size(); ti++) {
                    auto& t = tracks[ti];
                    if (t.matched) continue;

                    int missing = MAX(0, t.max_ttl - t.ttl - 1);
                    float dyn = DISTANCE_THRESHOLD + 22.0f * (float)missing;
                    if (dyn > 320.0f) dyn = 320.0f;

                    float dist = point_distance(atual, t.position);
                    int dd = abs(dmm_round - t.diameter_mm);
                    if (dist < dyn && dd <= 14 && dist < best_dist) {
                        best_dist = dist;
                        best_idx = ti;
                    }
                }

                OrangeTrack* tr = nullptr;
                if (best_idx >= 0) {
                    auto& t = tracks[best_idx];
                    t.position = atual;
                    t.diameter_mm = dmm_round;
                    t.ttl = t.max_ttl;
                    t.hits++;
                    t.matched = true;
                    tr = &t;
                    found = true;
                }

                if (!found) {
                    OrangeTrack t{};
                    t.position = atual;
                    t.diameter_mm = dmm_round;
                    t.ttl = MAX_TTL;
                    t.max_ttl = MAX_TTL;
                    t.hits = 1;
                    t.counted = false;
                    t.matched = true;
                    tracks.push_back(t);
                    tr = &tracks.back();
                }

                if (tr && !tr->counted && tr->hits >= MIN_HITS_TO_COUNT) {
                    int mem_idx = find_similar_counted_memory(counted_memory, tr->position, tr->diameter_mm, video.nframe);
                    if (mem_idx < 0) {
                        CountedOrange m{};
                        m.position = tr->position;
                        m.diameter_mm = tr->diameter_mm;
                        m.last_frame = video.nframe;
                        counted_memory.push_back(m);
                        total_laranjas++;
                    }
                    tr->counted = true;
                }

                unsigned char cb = 0, cg = 255, cr = 0;
                if (strcmp(calibre_ok, "NA TOLERANCIA") == 0) { cb = 0; cg = 255; cr = 255; }
                if (strcmp(calibre_ok, "NAO") == 0 || strcmp(categoria_ok, "NAO") == 0) { cb = 0; cg = 0; cr = 255; }

                cv::rectangle(frame,
                    cv::Point(blobs[i].x, blobs[i].y),
                    cv::Point(blobs[i].x + blobs[i].width, blobs[i].y + blobs[i].height),
                    cv::Scalar(cb, cg, cr), 2);
                cv::circle(frame, atual, 5, cv::Scalar(0, 0, 255), -1);

                std::string line1 = "LABEL:" + std::to_string(blobs[i].label) +
                    " AREA:" + std::to_string(blobs[i].area) +
                    " PERIMETRO:" + std::to_string(blobs[i].perimetro);
                std::string line2 = "CALIBRE:" + std::string(calibre) +
                    " DIAMETRO:" + std::to_string(dmm_round) +
                    "MM APROVADO:" + calibre_ok;
                std::string line3 = "DEFORMACAO:" + format_float(deform, 2) +
                    "% CATEGORIA:" + std::string(categoria) +
                    " APROVADO:" + categoria_ok;

                int tx = blobs[i].x;
                int ty = std::max(20, blobs[i].y - 10);
                draw_text(frame, line1, tx, ty, 0.5);
                draw_text(frame, line2, tx, ty + 18, 0.5);
                draw_text(frame, line3, tx, ty + 36, 0.5);
            }
        }

        display(frame, video, total_laranjas, laranjas_frame, diameters_frame_mm);

        cv::imshow("VC - VIDEO", frame);
        key = cv::waitKey(1);

        if (blobs != NULL) free(blobs);
    }

    vc_image_free(image);
    vc_image_free(bin);
    vc_image_free(labels);

    vc_timer();
    cv::destroyWindow("VC - VIDEO");
    capture.release();
    return 0;
}
