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
    <title>Admin - Hệ thống quản trị</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Admin Panel</div>
            <div class="muted">Quản lý dữ liệu và theo dõi hoạt động hệ thống</div>
        </div>
        <div class="nav-links">
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="index.php">Dashboard</a>
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin_drugs.php">Thuốc</a>
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin_diseases.php">Bệnh</a>
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin_links.php">Liên kết</a>
            <a class="btn" style="background: rgba(244, 63, 94, 0.16); box-shadow:none;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="admin-stats">
        <div class="glass-card stat-card"><div class="label">Tổng số thuốc</div><h2><?= number_format($stats['drugs']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng số bệnh</div><h2><?= number_format($stats['diseases']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng số protein</div><h2><?= number_format($stats['proteins']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tổng lượt dự đoán</div><h2><?= number_format($stats['predictions']) ?></h2></div>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <h3>Lượt dự đoán gần đây</h3>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Kiểu</th>
                        <th>Input</th>
                        <th>Kết quả</th>
                        <th>Thời gian</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($recent as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge" style="background: rgba(255,255,255,0.05);"><?= e((string) $row['query_type']) ?></span></td>
                            <td style="font-weight: 600;"><?= e((string) $row['input_text']) ?></td>
                            <td><span class="badge badge-drug">Top-<?= $row['top_k'] ?></span></td>
                            <td class="muted"><?= date('H:i d/m', strtotime((string)$row['created_at'])) ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="glass-card">
            <h3>Hệ thống dữ liệu</h3>
            <p class="muted" style="margin-bottom: 20px;">Quản lý các thực thể và liên kết sinh học trong CSDL.</p>
            <div style="padding: 20px; background: rgba(59, 130, 246, 0.1); border-radius: 18px; border: 1px solid rgba(59,130,246,0.2);">
                <div class="label" style="color: #93c5fd;">Tổng liên kết hiện tại</div>
                <div style="font-size: 2rem; font-weight: 800; color: #fff;"> <?= number_format($stats['links']) ?></div>
                <p class="muted" style="font-size: 12px; margin-top: 8px;">Dữ liệu này được sử dụng để xây dựng đồ thị cho mô hình HGT.</p>
            </div>
            <div style="margin-top: 22px; display: grid; gap: 12px;">
                <a href="admin_links.php" class="btn" style="width: 100%;">Quản lý liên kết ngay</a>
            </div>
        </div>
    </div>
</div>
</body>
</html>