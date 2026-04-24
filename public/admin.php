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
    <title>Admin Panel</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Admin Panel</div>
            <div class="muted">Quan ly du lieu va theo doi hoat dong he thong bang mot giao dien thong nhat.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="index.php">Dashboard</a>
            <a class="btn btn-ghost" href="admin_drugs.php">Thuoc</a>
            <a class="btn btn-ghost" href="admin_diseases.php">Benh</a>
            <a class="btn btn-ghost" href="admin_links.php">Lien ket</a>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <div class="admin-stats">
        <div class="glass-card stat-card"><div class="label">Tong so thuoc</div><h2><?= number_format($stats['drugs']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tong so benh</div><h2><?= number_format($stats['diseases']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tong so protein</div><h2><?= number_format($stats['proteins']) ?></h2></div>
        <div class="glass-card stat-card"><div class="label">Tong luot du doan</div><h2><?= number_format($stats['predictions']) ?></h2></div>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Luot du doan gan day</h3>
                    <p class="muted">Theo doi cac truy van moi nhat de kiem tra tinh on dinh cua he thong.</p>
                </div>
            </div>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Kieu</th>
                        <th>Input</th>
                        <th>Ket qua</th>
                        <th>Thoi gian</th>
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
                    <h3>He thong du lieu</h3>
                    <p class="muted">Thong ke tong quan giup theo doi quy mo du lieu su dung cho mo hinh.</p>
                </div>
            </div>

            <div class="surface-box">
                <div class="label">Tong lien ket hien tai</div>
                <div class="stat-card"><h2><?= number_format($stats['links']) ?></h2></div>
                <p class="muted">Du lieu lien ket nay duoc dung de xay dung do thi va luu cac quan he phuc vu huan luyen mo hinh HGT.</p>
            </div>

            <div class="grid section-spaced">
                <a href="admin_links.php" class="btn btn-full">Quan ly lien ket ngay</a>
            </div>
        </div>
    </div>
</div>
</body>
</html>
