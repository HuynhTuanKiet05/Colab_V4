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
        $stmt->execute([
            'source_code' => $sourceCode,
            'name' => $name,
            'smiles' => $smiles,
            'description' => $description
        ]);
        flash('success', 'Da them thuoc moi.');
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drugs WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Da xoa thuoc.');
    }

    redirect('admin_drugs.php');
}

$success = flash('success');
$rows = db()->query('SELECT * FROM drugs ORDER BY created_at DESC LIMIT 50')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Quan ly thuoc</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quan ly thuoc</div>
            <div class="muted">Them, xoa va theo doi cac thuc the thuoc trong co so du lieu.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="admin.php">Quay lai</a>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="grid split-panel">
        <div class="glass-card">
            <h3>Them thuc the thuoc moi</h3>
            <p class="muted spacer-lg">Dien thong tin dinh danh va dac trung hoa hoc co ban de bo sung cho du lieu quan tri.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div class="stack-tight">
                    <div class="form-group">
                        <label class="label">Ma nguon</label>
                        <input class="input" name="source_code" placeholder="Vi du: DB00014" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Ten thuoc</label>
                        <input class="input" name="name" placeholder="Vi du: Goserelin" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Cau truc SMILES</label>
                        <textarea class="input" name="smiles" placeholder="Danh cho thong tin cau truc phan tu..."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="label">Mo ta</label>
                        <textarea class="input" name="description" placeholder="Thong tin bo sung ve thuoc..."></textarea>
                    </div>
                    <button class="btn btn-full" type="submit">Luu thong tin</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Danh sach thuoc</h3>
                    <p class="muted">Hien thi toi da 50 ban ghi moi nhat.</p>
                </div>
                <div class="badge badge-drug"><?= count($rows) ?> records</div>
            </div>

            <div class="table-container table-scroll">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Ma nguon</th>
                        <th>Ten thuoc</th>
                        <th class="align-right">Hanh dong</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($rows)): ?>
                        <tr><td colspan="4" class="center-empty">Chua co du lieu thuoc.</td></tr>
                    <?php endif; ?>
                    <?php foreach ($rows as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge badge-drug"><?= e((string) $row['source_code']) ?></span></td>
                            <td><strong><?= e((string) $row['name']) ?></strong></td>
                            <td class="align-right">
                                <form method="post" class="inline-form" onsubmit="return confirm('Ban co chac chan muon xoa thuoc nay?');">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
                                    <button class="btn btn-danger btn-sm" type="submit">Xoa</button>
                                </form>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>
</body>
</html>
