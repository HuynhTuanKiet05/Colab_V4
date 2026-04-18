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

$recent = db()->query('SELECT * FROM prediction_requests ORDER BY created_at DESC LIMIT 10')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quản trị</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="brand">Trang quản trị</div>
        <div class="nav-links">
            <a class="btn btn-secondary" href="index.php">Dashboard</a>
            <a class="btn btn-secondary" href="admin_drugs.php">Quản lý thuốc</a>
            <a class="btn btn-secondary" href="admin_diseases.php">Quản lý bệnh</a>
            <a class="btn btn-secondary" href="admin_links.php">Quản lý liên kết</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="stats">
        <div class="stat"><div class="muted">Số thuốc</div><h2><?= $stats['drugs'] ?></h2></div>
        <div class="stat"><div class="muted">Số bệnh</div><h2><?= $stats['diseases'] ?></h2></div>
        <div class="stat"><div class="muted">Số protein</div><h2><?= $stats['proteins'] ?></h2></div>
        <div class="stat"><div class="muted">Lượt dự đoán</div><h2><?= $stats['predictions'] ?></h2></div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>Tổng liên kết thuốc - bệnh</h2>
        <p><strong><?= $stats['links'] ?></strong> liên kết trong CSDL.</p>
        <p class="muted">Bạn có thể thêm/sửa/xóa tại trang Quản lý liên kết.</p>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>Thống kê lượt dự đoán gần đây</h2>
        <table class="table">
            <thead>
            <tr>
                <th>ID</th>
                <th>Loại</th>
                <th>Input</th>
                <th>Top-k</th>
                <th>Thời gian</th>
            </tr>
            </thead>
            <tbody>
            <?php foreach ($recent as $row): ?>
                <tr>
                    <td><?= e((string) $row['id']) ?></td>
                    <td><?= e((string) $row['query_type']) ?></td>
                    <td><?= e((string) $row['input_text']) ?></td>
                    <td><?= e((string) $row['top_k']) ?></td>
                    <td><?= e((string) $row['created_at']) ?></td>
                </tr>
            <?php endforeach; ?>
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
