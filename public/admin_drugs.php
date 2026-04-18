<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? 'create';
    $sourceCode = trim($_POST['source_code'] ?? '');
    $name = trim($_POST['name'] ?? '');
    $smiles = trim($_POST['smiles'] ?? '');
    $description = trim($_POST['description'] ?? '');

    if ($action === 'create' && $sourceCode !== '' && $name !== '') {
        $stmt = db()->prepare('INSERT INTO drugs (source_code, name, smiles, description) VALUES (:source_code, :name, :smiles, :description)');
        $stmt->execute(compact('sourceCode', 'name', 'smiles', 'description') + [
            'source_code' => $sourceCode,
        ]);
        flash('success', 'Đã thêm thuốc mới.');
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drugs WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Đã xóa thuốc.');
    }

    redirect('admin_drugs.php');
}

$success = flash('success');
$rows = db()->query('SELECT * FROM drugs ORDER BY created_at DESC LIMIT 100')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quản lý thuốc</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="brand">Quản lý thuốc</div>
        <div class="nav-links">
            <a class="btn btn-secondary" href="admin.php">Quản trị</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="card">
        <h2>Thêm thuốc</h2>
        <form method="post">
            <input type="hidden" name="action" value="create">
            <div class="grid" style="grid-template-columns:1fr 1fr;">
                <input class="input" name="source_code" placeholder="Mã thuốc" required>
                <input class="input" name="name" placeholder="Tên thuốc" required>
            </div>
            <div style="margin-top:12px;"><textarea class="input" name="smiles" placeholder="SMILES"></textarea></div>
            <div style="margin-top:12px;"><textarea class="input" name="description" placeholder="Mô tả"></textarea></div>
            <div style="margin-top:12px;"><button class="btn" type="submit">Thêm thuốc</button></div>
        </form>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>Danh sách thuốc</h2>
        <table class="table">
            <thead><tr><th>ID</th><th>Mã</th><th>Tên</th><th>Hành động</th></tr></thead>
            <tbody>
            <?php foreach ($rows as $row): ?>
                <tr>
                    <td><?= e((string) $row['id']) ?></td>
                    <td><?= e((string) $row['source_code']) ?></td>
                    <td><?= e((string) $row['name']) ?></td>
                    <td>
                        <form method="post" onsubmit="return confirm('Xóa thuốc này?');">
                            <input type="hidden" name="action" value="delete">
                            <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
                            <button class="btn btn-danger" type="submit">Xóa</button>
                        </form>
                    </td>
                </tr>
            <?php endforeach; ?>
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
