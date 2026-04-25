<?php
require_once __DIR__ . '/../app/services/PredictionService.php';
require_login();

$user = current_user();
$allowedDatasets = ['B-dataset', 'C-dataset', 'F-dataset'];
$dataset = $_POST['dataset'] ?? 'C-dataset';
$dataset = in_array($dataset, $allowedDatasets, true) ? $dataset : 'C-dataset';
$topK = max(1, min(20, (int) ($_POST['top_k'] ?? 5)));
$drugInput = trim((string) ($_POST['drugs'] ?? ''));
$diseaseInput = trim((string) ($_POST['diseases'] ?? ''));
$resultData = null;
$error = null;
$apiHealthy = PredictionService::isApiHealthy();

if (empty($_SESSION['_csrf_compare_models'])) {
    $_SESSION['_csrf_compare_models'] = bin2hex(random_bytes(16));
}
$csrfToken = $_SESSION['_csrf_compare_models'];

function parse_compare_entities(string $raw): array
{
    $parts = preg_split('/[\r\n,;]+/', $raw) ?: [];
    $items = [];
    $seen = [];
    foreach ($parts as $part) {
        $value = trim($part);
        if ($value === '') {
            continue;
        }
        $key = strtolower($value);
        if (isset($seen[$key])) {
            continue;
        }
        $items[] = $value;
        $seen[$key] = true;
    }
    return $items;
}

function format_score(mixed $score): string
{
    return number_format((float) $score, 4);
}

function dataset_counts(string $dataset): array
{
    $base = realpath(__DIR__ . '/../AMDGT_original/data/' . $dataset);
    $countCsvRows = static function (?string $base, string $file): int {
        if ($base === null) {
            return 0;
        }
        $path = $base . DIRECTORY_SEPARATOR . $file;
        if (!is_file($path)) {
            return 0;
        }
        $handle = @fopen($path, 'r');
        if ($handle === false) {
            return 0;
        }
        $rows = 0;
        while (fgets($handle) !== false) {
            $rows++;
        }
        fclose($handle);
        return max(0, $rows - 1);
    };
    return [
        'drugs' => $countCsvRows($base ?: null, 'DrugFingerprint.csv'),
        'diseases' => $countCsvRows($base ?: null, 'DiseaseGIP.csv'),
        'pairs' => $countCsvRows($base ?: null, 'DrugDiseaseAssociationNumber.csv'),
    ];
}

function render_compare_graph(array $graph): string
{
    $nodes = $graph['nodes'] ?? [];
    $links = $graph['links'] ?? [];
    $drugs = [];
    $diseases = [];
    foreach ($nodes as $node) {
        if (($node['type'] ?? '') === 'drug') {
            $drugs[] = $node;
        } elseif (($node['type'] ?? '') === 'disease') {
            $diseases[] = $node;
        }
    }

    $rowCount = max(count($drugs), count($diseases), 1);
    $height = max(360, 110 + ($rowCount * 82));
    $leftX = 190;
    $rightX = 910;
    $topPad = 70;
    $bottomPad = 54;
    $usableHeight = $height - $topPad - $bottomPad;
    $drugPositions = [];
    $diseasePositions = [];

    foreach ($drugs as $index => $node) {
        $y = $topPad + (($index + 0.5) * $usableHeight / max(count($drugs), 1));
        $drugPositions[$node['id']] = [$leftX, $y, $node];
    }
    foreach ($diseases as $index => $node) {
        $y = $topPad + (($index + 0.5) * $usableHeight / max(count($diseases), 1));
        $diseasePositions[$node['id']] = [$rightX, $y, $node];
    }

    ob_start();
    ?>
    <svg class="compare-network" viewBox="0 0 1100 <?= e((string) $height) ?>" role="img" aria-label="Sơ đồ liên kết 2D thuốc - bệnh">
        <defs>
            <marker id="arrow-improved" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#198754"></path>
            </marker>
            <marker id="arrow-original" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"></path>
            </marker>
        </defs>
        <text x="110" y="34" class="compare-network-title">Thuốc</text>
        <text x="845" y="34" class="compare-network-title">Bệnh</text>
        <?php foreach ($links as $link): ?>
            <?php
            $source = $drugPositions[$link['source'] ?? ''] ?? null;
            $target = $diseasePositions[$link['target'] ?? ''] ?? null;
            if (!$source || !$target) {
                continue;
            }
            $delta = (float) ($link['delta'] ?? 0);
            $score = max(0.05, min(1.0, (float) ($link['improved_score'] ?? 0)));
            $stroke = $delta >= 0 ? '#198754' : '#d97706';
            $marker = $delta >= 0 ? 'url(#arrow-improved)' : 'url(#arrow-original)';
            $width = 1.5 + ($score * 5);
            $midX = ((float) $source[0] + (float) $target[0]) / 2;
            $midY = ((float) $source[1] + (float) $target[1]) / 2;
            ?>
            <path d="M <?= e((string) $source[0]) ?> <?= e((string) $source[1]) ?> C 410 <?= e((string) $source[1]) ?>, 690 <?= e((string) $target[1]) ?>, <?= e((string) $target[0]) ?> <?= e((string) $target[1]) ?>"
                  stroke="<?= e($stroke) ?>" stroke-width="<?= e(number_format($width, 2)) ?>" marker-end="<?= e($marker) ?>" class="compare-edge"></path>
            <g class="compare-edge-label">
                <rect x="<?= e((string) ($midX - 58)) ?>" y="<?= e((string) ($midY - 14)) ?>" width="116" height="28" rx="8"></rect>
                <text x="<?= e((string) $midX) ?>" y="<?= e((string) ($midY + 4)) ?>">I <?= e(format_score($link['improved_score'] ?? 0)) ?> / Δ <?= e(format_score($delta)) ?></text>
            </g>
        <?php endforeach; ?>

        <?php foreach ($drugPositions as $position): ?>
            <?php [$x, $y, $node] = $position; ?>
            <g class="compare-node compare-node-drug">
                <circle cx="<?= e((string) $x) ?>" cy="<?= e((string) $y) ?>" r="12"></circle>
                <rect x="<?= e((string) ($x - 150)) ?>" y="<?= e((string) ($y - 24)) ?>" width="132" height="48" rx="8"></rect>
                <text x="<?= e((string) ($x - 84)) ?>" y="<?= e((string) ($y - 3)) ?>"><?= e((string) ($node['label'] ?? $node['actual_id'] ?? 'Thuốc')) ?></text>
                <text x="<?= e((string) ($x - 84)) ?>" y="<?= e((string) ($y + 15)) ?>" class="compare-node-id"><?= e((string) ($node['actual_id'] ?? '')) ?></text>
            </g>
        <?php endforeach; ?>

        <?php foreach ($diseasePositions as $position): ?>
            <?php [$x, $y, $node] = $position; ?>
            <g class="compare-node compare-node-disease">
                <circle cx="<?= e((string) $x) ?>" cy="<?= e((string) $y) ?>" r="12"></circle>
                <rect x="<?= e((string) ($x + 18)) ?>" y="<?= e((string) ($y - 24)) ?>" width="150" height="48" rx="8"></rect>
                <text x="<?= e((string) ($x + 93)) ?>" y="<?= e((string) ($y - 3)) ?>"><?= e((string) ($node['label'] ?? $node['actual_id'] ?? 'Bệnh')) ?></text>
                <text x="<?= e((string) ($x + 93)) ?>" y="<?= e((string) ($y + 15)) ?>" class="compare-node-id"><?= e((string) ($node['actual_id'] ?? '')) ?></text>
            </g>
        <?php endforeach; ?>
    </svg>
    <?php
    return (string) ob_get_clean();
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $submittedToken = (string) ($_POST['csrf_token'] ?? '');
    $drugs = parse_compare_entities($drugInput);
    $diseases = parse_compare_entities($diseaseInput);

    if (!hash_equals($csrfToken, $submittedToken)) {
        $error = 'Phiên làm việc không hợp lệ. Vui lòng tải lại trang.';
    } elseif (!$apiHealthy) {
        $error = 'Python API đang ngoại tuyến. Vui lòng khởi động FastAPI ở cổng 8000.';
    } elseif (empty($drugs) || empty($diseases)) {
        $error = 'Vui lòng chọn ít nhất 1 thuốc và 1 bệnh.';
    } elseif (count($drugs) > 5 || count($diseases) > 5) {
        $error = 'Chỉ được chọn tối đa 5 thuốc và 5 bệnh trong một lần chẩn đoán.';
    } else {
        try {
            $resultData = PredictionService::comparePredict($drugs, $diseases, $topK, $dataset);
            PredictionService::saveComparisonHistory((int) $user['id'], $dataset, $drugs, $diseases, $topK, $resultData);
            flash('success', 'Đã chạy so sánh 2 mô hình thành công.');
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$counts = dataset_counts($dataset);
$comparisonRows = $resultData['comparison'] ?? [];
$chartRows = array_slice($comparisonRows, 0, 10);
$chartLabels = array_map(
    fn ($row) => ($row['drug_id'] ?? '') . ' / ' . ($row['disease_id'] ?? ''),
    $chartRows
);
$chartOriginal = array_map(fn ($row) => (float) ($row['original_score'] ?? 0), $chartRows);
$chartImproved = array_map(fn ($row) => (float) ($row['improved_score'] ?? 0), $chartRows);
$chartDelta = array_map(fn ($row) => (float) ($row['delta'] ?? 0), $chartRows);
$pairCount = count($comparisonRows);
$improvedWins = 0;
$originalWins = 0;
$totalDelta = 0.0;
$topImprovedRow = null;
foreach ($comparisonRows as $row) {
    $winner = (string) ($row['winner'] ?? 'tie');
    if ($winner === 'improved') {
        $improvedWins++;
    } elseif ($winner === 'original') {
        $originalWins++;
    }
    $totalDelta += (float) ($row['delta'] ?? 0);
    if ($topImprovedRow === null || (float) ($row['improved_score'] ?? 0) > (float) ($topImprovedRow['improved_score'] ?? 0)) {
        $topImprovedRow = $row;
    }
}
$averageDelta = $pairCount > 0 ? $totalDelta / $pairCount : 0.0;
$leadRate = $pairCount > 0 ? ($improvedWins / $pairCount) * 100 : 0.0;
$topImprovedPair = $topImprovedRow
    ? (($topImprovedRow['drug_id'] ?? '') . ' / ' . ($topImprovedRow['disease_id'] ?? ''))
    : 'Chưa có dữ liệu';
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMNTDDA AI · Chẩn đoán liên kết thuốc - bệnh</title>
    <link rel="stylesheet" href="assets/style.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght@400;500;700&display=swap">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div id="loader" class="loading-overlay">Đang chạy 2 mô hình trên cùng tập đầu vào...</div>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">AMNTDDA AI</div>
            <div class="muted">Nền tảng dự đoán &amp; so sánh liên kết Thuốc – Bệnh trên đồ thị HGT.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-sm" href="#compare-form-panel">Bắt đầu chẩn đoán</a>
            <a class="btn btn-ghost btn-sm" href="history.php">Lịch sử</a>
            <?php if (($user['role'] ?? '') === 'admin'): ?>
                <a class="btn btn-ghost btn-sm" href="admin.php">Quản trị</a>
            <?php endif; ?>
            <a class="btn btn-danger btn-sm" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="app-shell">
        <aside class="side-nav glass-card">
            <div class="side-nav-title-wrap">
                <div class="side-nav-title">AMNTDDA</div>
                <div class="side-nav-meta">Precision Medical AI</div>
            </div>
            <div class="side-nav-menu">
                <a class="side-nav-item side-nav-item-active" href="#overview"><span class="material-symbols-outlined">dashboard</span><span>Tổng quan</span></a>
                <a class="side-nav-item" href="#compare-form-panel"><span class="material-symbols-outlined">tune</span><span>Chẩn đoán</span></a>
                <a class="side-nav-item" href="<?= $resultData ? '#result-overview' : '#quick-start' ?>"><span class="material-symbols-outlined"><?= $resultData ? 'insights' : 'menu_book' ?></span><span><?= $resultData ? 'Kết quả' : 'Hướng dẫn' ?></span></a>
                <a class="side-nav-item" href="history.php"><span class="material-symbols-outlined">history</span><span>Lịch sử</span></a>
                <?php if (($user['role'] ?? '') === 'admin'): ?>
                    <a class="side-nav-item" href="admin.php"><span class="material-symbols-outlined">admin_panel_settings</span><span>Quản trị</span></a>
                <?php endif; ?>
            </div>
        </aside>
        <div class="main-shell">

    <div class="glass-card hero-banner" id="overview">
        <div class="hero-grid">
            <div class="hero-pitch">
                <span class="badge badge-drug">AI Graph Prediction · MVP</span>
                <h1>Bảng điều khiển chẩn đoán Thuốc – Bệnh gọn, rõ và sẵn sàng để demo.</h1>
                <p class="muted">Một màn hình duy nhất để chọn dữ liệu, chạy song song mô hình gốc và mô hình cải tiến, rồi đọc kết quả bằng bảng điểm, delta và sơ đồ liên kết.</p>
                <div class="hero-actions">
                    <a class="btn" href="#compare-form-panel">Bắt đầu chẩn đoán</a>
                    <a class="btn btn-ghost" href="<?= $resultData ? '#result-overview' : '#quick-start' ?>"><?= $resultData ? 'Xem kết quả hiện tại' : 'Xem quy trình sử dụng' ?></a>
                </div>
                <div class="hero-bullets">
                    <div class="hero-bullet">So sánh trực tiếp mô hình gốc và mô hình cải tiến trên cùng đầu vào.</div>
                    <div class="hero-bullet">Làm việc với 3 bộ dữ liệu chuẩn: B-dataset, C-dataset và F-dataset.</div>
                    <div class="hero-bullet">Đọc kết quả theo ba lớp: bảng điểm, delta và sơ đồ liên kết 2D.</div>
                </div>
            </div>
            <div class="status-card">
                <div class="label">Trạng thái hệ thống</div>
                <span class="badge <?= $apiHealthy ? 'badge-success' : 'badge-neutral' ?>">
                    <?= $apiHealthy ? 'AI API · Trực tuyến' : 'AI API · Ngoại tuyến' ?>
                </span>
                <p class="muted"><?= $apiHealthy ? 'Server FastAPI sẵn sàng phục vụ chẩn đoán.' : 'Hãy khởi động Python API ở cổng 8000.' ?></p>
                <div class="status-card-row">
                    <span>Bộ dữ liệu hiện hành</span>
                    <span><?= e($dataset) ?></span>
                </div>
                <div class="status-card-row">
                    <span>Phiên bản mô hình</span>
                    <span>HGT + MVA · Improved</span>
                </div>
                <div class="status-card-row">
                    <span>Top-K mặc định</span>
                    <span><?= e((string) $topK) ?></span>
                </div>
                <div class="status-mini-grid">
                    <div class="status-mini-tile">
                        <strong>3</strong>
                        <span>Bộ dữ liệu</span>
                    </div>
                    <div class="status-mini-tile">
                        <strong>2</strong>
                        <span>Mô hình</span>
                    </div>
                    <div class="status-mini-tile">
                        <strong>1–20</strong>
                        <span>Top-K</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <?php if (!$apiHealthy): ?>
        <div class="alert alert-error">Cảnh báo: Python AI API ở cổng 8000 đang ngắt kết nối. Vui lòng khởi động trước khi chạy chẩn đoán.</div>
    <?php endif; ?>
    <?php if ($error): ?>
        <div class="alert alert-error"><?= e($error) ?></div>
    <?php endif; ?>
    <?php if ($success): ?>
        <div class="alert alert-success"><?= e($success) ?></div>
    <?php endif; ?>

    <div class="stats-row">
        <div class="stat-tile stat-tile-drug">
            <div class="stat-tile-label">Tổng số thuốc</div>
            <div class="stat-tile-value"><?= number_format($counts['drugs']) ?></div>
            <div class="stat-tile-sub">Thực thể thuốc trong <?= e($dataset) ?></div>
        </div>
        <div class="stat-tile stat-tile-disease">
            <div class="stat-tile-label">Tổng số bệnh</div>
            <div class="stat-tile-value"><?= number_format($counts['diseases']) ?></div>
            <div class="stat-tile-sub">Thực thể bệnh trong <?= e($dataset) ?></div>
        </div>
        <div class="stat-tile stat-tile-link">
            <div class="stat-tile-label">Cặp liên kết đã biết</div>
            <div class="stat-tile-value"><?= number_format($counts['pairs']) ?></div>
            <div class="stat-tile-sub">Số cặp Thuốc – Bệnh có nhãn dương trong dữ liệu huấn luyện.</div>
        </div>
        <div class="stat-tile stat-tile-model">
            <div class="stat-tile-label">Mô hình triển khai</div>
            <div class="stat-tile-value">2</div>
            <div class="stat-tile-sub">Mô hình gốc &amp; mô hình cải tiến cùng chạy trực tiếp trên web.</div>
            <span class="stat-tile-pill <?= $apiHealthy ? 'pill-success' : 'pill-warn' ?>"><?= $apiHealthy ? 'Sẵn sàng' : 'Tạm dừng' ?></span>
        </div>
    </div>

    <div class="glass-card search-container" id="compare-form-panel">
        <div class="section-header">
            <div>
                <h2>Advanced Configuration Interface</h2>
                <p class="muted">Thiết lập tham số tìm kiếm và so sánh mô hình.</p>
            </div>
            <span class="badge <?= $apiHealthy ? 'badge-success' : 'badge-neutral' ?>"><?= $apiHealthy ? 'API trực tuyến' : 'API ngoại tuyến' ?></span>
        </div>

        <div class="diagnosis-layout">
            <form method="post" id="compare-form" class="diagnosis-main" onsubmit="document.getElementById('loader').style.display='grid'">
                <input type="hidden" name="csrf_token" value="<?= e($csrfToken) ?>">
                <input type="hidden" name="drugs" id="drugs-hidden" value="<?= e($drugInput) ?>">
                <input type="hidden" name="diseases" id="diseases-hidden" value="<?= e($diseaseInput) ?>">
                <div class="compare-form-grid">
                    <div class="compare-form-row">
                        <div class="form-group">
                            <label class="label">Bộ dữ liệu</label>
                            <select class="select" name="dataset" id="dataset-select">
                                <?php foreach ($allowedDatasets as $option): ?>
                                    <option value="<?= e($option) ?>" <?= $dataset === $option ? 'selected' : '' ?>><?= e($option) ?></option>
                                <?php endforeach; ?>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="label">Top-K kết quả</label>
                            <input class="input" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="label">Danh sách thuốc · tối đa 5</label>
                        <div class="entity-picker" id="drug-picker">
                            <div class="entity-picker-tags" id="drug-tags"></div>
                            <input type="text" class="entity-picker-search" id="drug-search" placeholder="Tìm kiếm thuốc theo tên hoặc mã ID..." autocomplete="off">
                            <div class="entity-picker-list" id="drug-list"><div class="entity-picker-msg">Đang tải dữ liệu...</div></div>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="label">Danh sách bệnh · tối đa 5</label>
                        <div class="entity-picker" id="disease-picker">
                            <div class="entity-picker-tags" id="disease-tags"></div>
                            <input type="text" class="entity-picker-search" id="disease-search" placeholder="Tìm kiếm bệnh theo tên hoặc mã ID..." autocomplete="off">
                            <div class="entity-picker-list" id="disease-list"><div class="entity-picker-msg">Đang tải dữ liệu...</div></div>
                        </div>
                    </div>
                    <div class="form-group compare-action">
                        <button class="btn btn-full" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Chạy chẩn đoán &amp; so sánh 2 mô hình</button>
                    </div>
                </div>
            </form>
            <div class="diagnosis-aside">
                <div class="diagnosis-panel">
                    <span class="label">Phiên chạy hiện tại</span>
                    <div class="diagnosis-kv">
                        <span>Dataset</span>
                        <strong><?= e($dataset) ?></strong>
                    </div>
                    <div class="diagnosis-kv">
                        <span>Giới hạn đầu vào</span>
                        <strong>5 thuốc / 5 bệnh</strong>
                    </div>
                    <div class="diagnosis-kv">
                        <span>Top-K hiện tại</span>
                        <strong><?= e((string) $topK) ?></strong>
                    </div>
                    <div class="diagnosis-kv">
                        <span>API</span>
                        <strong><?= $apiHealthy ? 'Trực tuyến' : 'Ngoại tuyến' ?></strong>
                    </div>
                </div>
                <div class="diagnosis-panel">
                    <span class="label">Bạn sẽ nhận được</span>
                    <div class="diagnosis-list">
                        <div class="diagnosis-list-item">
                            <strong>Hai bảng kết quả song song</strong>
                            <span>Mô hình gốc và mô hình cải tiến hiển thị cạnh nhau để đọc nhanh.</span>
                        </div>
                        <div class="diagnosis-list-item">
                            <strong>Bảng delta để so sánh chất lượng</strong>
                            <span>Xem chênh lệch điểm và mô hình thắng cho từng cặp Thuốc – Bệnh.</span>
                        </div>
                        <div class="diagnosis-list-item">
                            <strong>Biểu đồ trực quan cho phần trình bày</strong>
                            <span>Biểu đồ cột và sơ đồ 2D giúp giải thích kết quả rõ ràng hơn.</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <?php if ($resultData): ?>
        <div class="stats-row results-summary-row section-spaced" id="result-overview">
            <div class="stat-tile stat-tile-link">
                <div class="stat-tile-label">Cặp đã phân tích</div>
                <div class="stat-tile-value"><?= number_format($pairCount) ?></div>
                <div class="stat-tile-sub">Tổng số cặp Thuốc – Bệnh được xử lý trong lần chạy này.</div>
            </div>
            <div class="stat-tile stat-tile-model">
                <div class="stat-tile-label">Cải tiến dẫn trước</div>
                <div class="stat-tile-value"><?= number_format($improvedWins) ?></div>
                <div class="stat-tile-sub"><?= $pairCount > 0 ? e(number_format($leadRate, 1)) . '% số cặp có điểm cao hơn mô hình gốc.' : 'Chưa có dữ liệu để tính tỷ lệ.' ?></div>
            </div>
            <div class="stat-tile <?= $averageDelta >= 0 ? 'stat-tile-drug' : 'stat-tile-disease' ?>">
                <div class="stat-tile-label">Delta trung bình</div>
                <div class="stat-tile-value"><?= e(format_score($averageDelta)) ?></div>
                <div class="stat-tile-sub"><?= $averageDelta >= 0 ? 'Giá trị dương cho thấy xu hướng cải thiện tổng thể.' : 'Giá trị âm cho thấy mô hình gốc đang nhỉnh hơn trong lần chạy này.' ?></div>
            </div>
            <div class="stat-tile stat-tile-drug">
                <div class="stat-tile-label">Điểm cải tiến cao nhất</div>
                <div class="stat-tile-value"><?= e(format_score($topImprovedRow['improved_score'] ?? 0)) ?></div>
                <div class="stat-tile-sub"><?= e($topImprovedPair) ?></div>
            </div>
        </div>

        <div class="grid grid-2-equal">
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Mô hình gốc</h3>
                        <p class="muted mono"><?= e((string) ($resultData['models']['original']['checkpoint'] ?? '')) ?></p>
                    </div>
                    <span class="badge badge-drug"><?= count($comparisonRows) ?> cặp</span>
                </div>
                <div class="table-container">
                    <table class="table">
                        <thead><tr><th>Thuốc</th><th>Bệnh</th><th>Điểm</th></tr></thead>
                        <tbody>
                        <?php foreach (($resultData['models']['original']['results'] ?? []) as $row): ?>
                            <tr>
                                <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                                <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                                <td class="score-text"><?= e(format_score($row['score'] ?? 0)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Mô hình cải tiến</h3>
                        <p class="muted mono"><?= e((string) ($resultData['models']['improved']['checkpoint'] ?? '')) ?></p>
                    </div>
                    <span class="badge badge-success">Improved</span>
                </div>
                <div class="table-container">
                    <table class="table">
                        <thead><tr><th>Thuốc</th><th>Bệnh</th><th>Điểm</th></tr></thead>
                        <tbody>
                        <?php foreach (($resultData['models']['improved']['results'] ?? []) as $row): ?>
                            <tr>
                                <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                                <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                                <td class="score-text"><?= e(format_score($row['score'] ?? 0)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="glass-card section-spaced">
            <div class="section-header">
                <div>
                    <h3>Bảng so sánh delta</h3>
                    <p class="muted">Delta = Điểm cải tiến − Điểm gốc. Mô hình thắng được tô đậm.</p>
                </div>
                <span class="badge badge-drug"><?= count($comparisonRows) ?> cặp</span>
            </div>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>Thuốc</th>
                        <th>Bệnh</th>
                        <th>Gốc</th>
                        <th>Cải tiến</th>
                        <th>Delta</th>
                        <th>Mô hình thắng</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($comparisonRows as $row): ?>
                        <tr>
                            <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                            <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                            <td class="score-text"><?= e(format_score($row['original_score'] ?? 0)) ?></td>
                            <td class="score-text"><?= e(format_score($row['improved_score'] ?? 0)) ?></td>
                            <td class="<?= ((float) ($row['delta'] ?? 0)) >= 0 ? 'delta-positive' : 'delta-negative' ?>"><?= e(format_score($row['delta'] ?? 0)) ?></td>
                            <td>
                                <?php
                                $winner = (string) ($row['winner'] ?? 'tie');
                                $winnerLabel = $winner === 'improved' ? 'Cải tiến' : ($winner === 'original' ? 'Gốc' : 'Hòa');
                                $winnerClass = $winner === 'improved' ? 'badge-success' : 'badge-neutral';
                                ?>
                                <span class="badge <?= e($winnerClass) ?>"><?= e($winnerLabel) ?></span>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="grid grid-2-equal section-spaced compare-visual-grid">
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Biểu đồ cột so sánh</h3>
                        <p class="muted">Hiển thị tối đa 10 cặp theo điểm của mô hình cải tiến.</p>
                    </div>
                </div>
                <div class="chart-panel">
                    <canvas id="comparisonChart"></canvas>
                    <div id="chartFallback" class="chart-fallback">Không tải được Chart.js. Vui lòng kiểm tra kết nối mạng hoặc tham khảo bảng so sánh phía trên.</div>
                </div>
            </div>
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Sơ đồ liên kết 2D</h3>
                        <p class="muted">Cột trái: thuốc · cột phải: bệnh · cạnh: kết quả của mô hình cải tiến.</p>
                    </div>
                </div>
                <?= render_compare_graph($resultData['graph2d'] ?? []) ?>
            </div>
        </div>
    <?php else: ?>
        <div class="glass-card section-spaced" id="quick-start">
            <div class="section-header">
                <div>
                    <h3>Sẵn sàng cho phiên chẩn đoán đầu tiên</h3>
                    <p class="muted">Quy trình gọn để thao tác nhanh và trình bày rõ ràng trong demo.</p>
                </div>
            </div>
            <div class="quick-start-grid">
                <div class="quick-start-card">
                    <span class="step-pill">Bước 1</span>
                    <h4>Chọn bộ dữ liệu</h4>
                    <p>Bắt đầu với C-dataset nếu bạn cần bộ dữ liệu cân bằng và quen thuộc cho demo.</p>
                </div>
                <div class="quick-start-card">
                    <span class="step-pill">Bước 2</span>
                    <h4>Chọn thuốc và bệnh</h4>
                    <p>Dùng bộ lọc tìm kiếm để chọn ít nhất 1 thuốc và 1 bệnh, tối đa 5 + 5 cho mỗi lần chạy.</p>
                </div>
                <div class="quick-start-card">
                    <span class="step-pill">Bước 3</span>
                    <h4>Đọc kết quả theo từng lớp</h4>
                    <p>Xem bảng điểm, delta và hai biểu đồ để kể câu chuyện kết quả một cách mạch lạc.</p>
                </div>
            </div>
        </div>
    <?php endif; ?>
        </div>
    </div>
</div>

<?php if ($resultData): ?>
    <script>
    const chartLabels = <?= json_encode($chartLabels, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
    const originalScores = <?= json_encode($chartOriginal, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
    const improvedScores = <?= json_encode($chartImproved, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
    const deltas = <?= json_encode($chartDelta, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
    const chartElement = document.getElementById('comparisonChart');
    if (chartElement && window.Chart) {
        new Chart(chartElement, {
            type: 'bar',
            data: {
                labels: chartLabels,
                datasets: [
                    { label: 'Mô hình gốc', data: originalScores, backgroundColor: 'rgba(148, 163, 184, 0.55)', borderColor: 'rgba(226, 232, 240, 0.55)', borderWidth: 1, borderRadius: 8 },
                    { label: 'Mô hình cải tiến', data: improvedScores, backgroundColor: 'rgba(99, 102, 241, 0.78)', borderColor: 'rgba(168, 85, 247, 0.72)', borderWidth: 1, borderRadius: 8 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#d8deea',
                            usePointStyle: true,
                            boxWidth: 10,
                            boxHeight: 10,
                            padding: 18
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 12, 18, 0.92)',
                        titleColor: '#f8fafc',
                        bodyColor: '#d8deea',
                        borderColor: 'rgba(255, 255, 255, 0.08)',
                        borderWidth: 1,
                        callbacks: {
                            afterBody: (items) => {
                                const index = items[0]?.dataIndex ?? 0;
                                return `Delta: ${Number(deltas[index] || 0).toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: { color: '#aeb7c8' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' }
                    },
                    x: {
                        ticks: { color: '#aeb7c8', maxRotation: 45, minRotation: 0 },
                        grid: { color: 'rgba(255, 255, 255, 0.04)' }
                    }
                }
            }
        });
    } else {
        const fallback = document.getElementById('chartFallback');
        if (fallback) {
            fallback.style.display = 'grid';
        }
    }
    </script>
<?php endif; ?>

<script>
(() => {
    const API = <?= json_encode(rtrim((string) config('python_api.base_url'), '/'), JSON_UNESCAPED_SLASHES) ?>;
    const MAX = 5;
    const preselDrugs = <?= json_encode(parse_compare_entities($drugInput), JSON_UNESCAPED_UNICODE) ?>;
    const preselDiseases = <?= json_encode(parse_compare_entities($diseaseInput), JSON_UNESCAPED_UNICODE) ?>;

    const state = {
        drug: { all: [], selected: new Set(), el: { tags: 'drug-tags', list: 'drug-list', search: 'drug-search', hidden: 'drugs-hidden' } },
        disease: { all: [], selected: new Set(), el: { tags: 'disease-tags', list: 'disease-list', search: 'disease-search', hidden: 'diseases-hidden' } }
    };

    function esc(t) { const d = document.createElement('span'); d.textContent = t; return d.innerHTML; }

    function render(type) {
        const s = state[type];
        const q = document.getElementById(s.el.search).value.toLowerCase();
        const listEl = document.getElementById(s.el.list);
        const filtered = s.all.filter(i => !q || `${i.id} ${i.name}`.toLowerCase().includes(q));

        if (!s.all.length) { listEl.innerHTML = '<div class="entity-picker-msg">Chưa có dữ liệu</div>'; renderTags(type); return; }
        if (!filtered.length) { listEl.innerHTML = '<div class="entity-picker-msg">Không tìm thấy</div>'; renderTags(type); return; }

        listEl.innerHTML = filtered.map(i => {
            const sel = s.selected.has(i.id);
            const dis = !sel && s.selected.size >= MAX;
            return `<div class="entity-picker-item${sel ? ' ep-selected' : ''}${dis ? ' ep-disabled' : ''}" data-id="${esc(i.id)}">` +
                `<input type="checkbox"${sel ? ' checked' : ''}${dis ? ' disabled' : ''}>` +
                `<span class="entity-picker-item-name">${esc(i.name)}</span>` +
                `<span class="entity-picker-id">${esc(i.id)}</span></div>`;
        }).join('') + `<div class="entity-picker-count">Đã chọn ${s.selected.size}/${MAX}</div>`;

        listEl.querySelectorAll('.entity-picker-item').forEach(el => {
            el.addEventListener('click', () => {
                const id = el.dataset.id;
                if (s.selected.has(id)) { s.selected.delete(id); }
                else if (s.selected.size < MAX) { s.selected.add(id); }
                else { return; }
                sync(type); render(type);
            });
        });
        renderTags(type);
    }

    function renderTags(type) {
        const s = state[type];
        const cls = type === 'drug' ? 'entity-tag-drug' : 'entity-tag-disease';
        const el = document.getElementById(s.el.tags);
        el.innerHTML = Array.from(s.selected).map(id => {
            const item = s.all.find(i => i.id === id);
            return `<span class="entity-tag ${cls}" data-id="${esc(id)}">${esc(item ? item.name : id)} <span class="entity-tag-remove">×</span></span>`;
        }).join('');
        el.querySelectorAll('.entity-tag').forEach(tag => {
            tag.addEventListener('click', () => { s.selected.delete(tag.dataset.id); sync(type); render(type); });
        });
    }

    function sync(type) {
        document.getElementById(state[type].el.hidden).value = Array.from(state[type].selected).join(',');
    }

    async function load(dataset) {
        for (const type of ['drug', 'disease']) {
            document.getElementById(state[type].el.list).innerHTML = '<div class="entity-picker-msg">Đang tải...</div>';
            document.getElementById(state[type].el.search).value = '';
        }
        try {
            const res = await fetch(`${API}/entities?dataset=${encodeURIComponent(dataset)}`);
            if (!res.ok) throw new Error();
            const data = await res.json();
            state.drug.all = data.drugs || [];
            state.disease.all = data.diseases || [];
        } catch { state.drug.all = []; state.disease.all = []; }
        render('drug'); render('disease');
    }

    function preselect(type, values) {
        const s = state[type];
        values.forEach(v => {
            const vl = v.toLowerCase();
            const m = s.all.find(i => i.id.toLowerCase() === vl || i.name.toLowerCase() === vl);
            if (m && s.selected.size < MAX) s.selected.add(m.id);
        });
        sync(type);
    }

    const sel = document.getElementById('dataset-select');
    sel.addEventListener('change', () => { state.drug.selected.clear(); state.disease.selected.clear(); load(sel.value); });
    document.getElementById('drug-search').addEventListener('input', () => render('drug'));
    document.getElementById('disease-search').addEventListener('input', () => render('disease'));

    load(sel.value).then(() => {
        preselect('drug', preselDrugs);
        preselect('disease', preselDiseases);
        render('drug'); render('disease');
    });
})();
</script>
</body>
</html>
