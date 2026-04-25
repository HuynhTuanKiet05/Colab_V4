<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

$stats = [
    'drugs' => (int) db()->query('SELECT COUNT(*) FROM drugs')->fetchColumn(),
    'diseases' => (int) db()->query('SELECT COUNT(*) FROM diseases')->fetchColumn(),
    'proteins' => (int) db()->query('SELECT COUNT(*) FROM proteins')->fetchColumn(),
    'predictions' => (int) db()->query('SELECT COUNT(*) FROM prediction_requests')->fetchColumn(),
    'links' => (int) db()->query('SELECT COUNT(*) FROM drug_disease_links')->fetchColumn(),
];

$recent = db()->query('SELECT * FROM prediction_requests ORDER BY created_at DESC LIMIT 8')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Khu vực quản trị · AMNTDDA AI</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Khu vực quản trị</div>
            <div class="muted">Quản lý dữ liệu và theo dõi hoạt động hệ thống AMNTDDA AI trong một giao diện thống nhất.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="index.php">Trang chủ</a>
            <a class="btn btn-ghost" href="admin_drugs.php">Thuốc</a>
            <a class="btn btn-ghost" href="admin_diseases.php">Bệnh</a>
            <a class="btn btn-ghost" href="admin_links.php">Liên kết</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="admin-stats">
        <div class="glass-card stat-card"><div class="label">Tổng số thuốc</div><h2><?= number_format($stats['drugs']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng số bệnh</div><h2><?= number_format($stats['diseases']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng số protein</div><h2><?= number_format($stats['proteins']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng lượt chẩn đoán</div><h2><?= number_format($stats['predictions']) ?></h2></div>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Lượt chẩn đoán gần đây</h3>
                    <p class="muted">Theo dõi các truy vấn mới nhất để kiểm tra tính ổn định của hệ thống.</p>
                </div>
            </div>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Kiểu</th>
                        <th>Truy vấn</th>
                        <th>Kết quả</th>
                        <th>Thời gian</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($recent as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge badge-neutral"><?= e((string) $row['query_type']) ?></span></td>
                            <td><strong><?= e((string) $row['input_text']) ?></strong></td>
                            <td><span class="badge badge-drug">Top-<?= $row['top_k'] ?></span></td>
                            <td class="muted"><?= date('H:i d/m', strtotime((string) $row['created_at'])) ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Hệ thống dữ liệu</h3>
                    <p class="muted">Thống kê tổng quan giúp theo dõi quy mô dữ liệu sử dụng cho mô hình.</p>
                </div>
            </div>

            <div class="surface-box">
                <div class="label">Tổng liên kết hiện tại</div>
                <div class="stat-card"><h2><?= number_format($stats['links']) ?></h2></div>
                <p class="muted">Dữ liệu liên kết dùng để xây dựng đồ thị và lưu quan hệ phục vụ huấn luyện mô hình HGT.</p>
            </div>

            <div class="grid section-spaced">
                <a href="admin_links.php" class="btn btn-full">Quản lý liên kết ngay</a>
            </div>
        </div>
    </div>
</div>
</body>
</html>
