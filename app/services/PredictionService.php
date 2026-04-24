<?php

declare(strict_types=1);

require_once __DIR__ . '/../bootstrap.php';

class PredictionService
{
    public static function isApiHealthy(): bool
    {
        $url = rtrim((string) config('python_api.base_url'), '/') . '/health';
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 5,
        ]);

        $response = curl_exec($ch);
        $error = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false || $error || $statusCode >= 400) {
            return false;
        }

        $decoded = json_decode($response, true);
        return is_array($decoded) && ($decoded['status'] ?? null) === 'ok';
    }

    public static function callPythonApi(string $queryType, string $inputText, int $topK = 10, string $dataset = 'C-dataset'): array
    {
        $url = rtrim((string) config('python_api.base_url'), '/') . '/predict';
        $payload = json_encode([
            'query_type' => $queryType,
            'input_text' => $inputText,
            'top_k' => $topK,
            'dataset' => $dataset,
        ], JSON_UNESCAPED_UNICODE);

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => $payload,
            CURLOPT_TIMEOUT => (int) config('python_api.timeout', 20),
        ]);

        $response = curl_exec($ch);
        $error = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false || $error) {
            throw new RuntimeException('Không thể kết nối Python API: ' . $error);
        }

        if ($statusCode >= 400) {
            throw new RuntimeException('Python API trả lỗi HTTP ' . $statusCode);
        }

        $decoded = json_decode($response, true);
        if (!is_array($decoded)) {
            throw new RuntimeException('Dữ liệu trả về từ Python API không hợp lệ.');
        }

        return $decoded;
    }

    public static function comparePredict(array $drugs, array $diseases, int $topK = 5, string $dataset = 'C-dataset'): array
    {
        $url = rtrim((string) config('python_api.base_url'), '/') . '/compare_predict';
        $payload = json_encode([
            'dataset' => $dataset,
            'drugs' => array_values($drugs),
            'diseases' => array_values($diseases),
            'top_k' => $topK,
        ], JSON_UNESCAPED_UNICODE);

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => $payload,
            CURLOPT_TIMEOUT => max(60, (int) config('python_api.compare_timeout', 180)),
        ]);

        $response = curl_exec($ch);
        $error = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false || $error) {
            throw new RuntimeException('Khong the ket noi Python API: ' . $error);
        }

        $decoded = json_decode($response, true);
        if ($statusCode >= 400) {
            $detail = is_array($decoded) ? ($decoded['detail'] ?? null) : null;
            throw new RuntimeException(is_string($detail) ? $detail : ('Python API tra loi HTTP ' . $statusCode));
        }

        if (!is_array($decoded)) {
            throw new RuntimeException('Du lieu tra ve tu Python API khong hop le.');
        }

        return $decoded;
    }

    public static function saveComparisonHistory(int $userId, string $dataset, array $drugs, array $diseases, int $topK, array $apiResult): ?int
    {
        try {
            $stmt = db()->prepare(
                'INSERT INTO model_comparison_requests (user_id, dataset, drugs_json, diseases_json, top_k, status, request_ip, result_json)
                 VALUES (:user_id, :dataset, :drugs_json, :diseases_json, :top_k, :status, :request_ip, :result_json)'
            );
            $stmt->execute([
                'user_id' => $userId,
                'dataset' => $dataset,
                'drugs_json' => json_encode(array_values($drugs), JSON_UNESCAPED_UNICODE),
                'diseases_json' => json_encode(array_values($diseases), JSON_UNESCAPED_UNICODE),
                'top_k' => $topK,
                'status' => 'success',
                'request_ip' => $_SERVER['REMOTE_ADDR'] ?? '127.0.0.1',
                'result_json' => json_encode($apiResult, JSON_UNESCAPED_UNICODE),
            ]);

            $requestId = (int) db()->lastInsertId();
            $resultStmt = db()->prepare(
                'INSERT INTO model_comparison_results
                    (request_id, rank_no, drug_code, drug_name, disease_code, disease_name, original_score, improved_score, delta_score, winner, metadata_json)
                 VALUES
                    (:request_id, :rank_no, :drug_code, :drug_name, :disease_code, :disease_name, :original_score, :improved_score, :delta_score, :winner, :metadata_json)'
            );

            foreach (($apiResult['comparison'] ?? []) as $index => $row) {
                $resultStmt->execute([
                    'request_id' => $requestId,
                    'rank_no' => $index + 1,
                    'drug_code' => $row['drug_id'] ?? '',
                    'drug_name' => $row['drug_name'] ?? '',
                    'disease_code' => $row['disease_id'] ?? '',
                    'disease_name' => $row['disease_name'] ?? '',
                    'original_score' => $row['original_score'] ?? 0,
                    'improved_score' => $row['improved_score'] ?? 0,
                    'delta_score' => $row['delta'] ?? 0,
                    'winner' => $row['winner'] ?? 'tie',
                    'metadata_json' => json_encode($row, JSON_UNESCAPED_UNICODE),
                ]);
            }

            return $requestId;
        } catch (PDOException $e) {
            return null;
        }
    }

    public static function saveHistory(int $userId, string $queryType, string $inputText, int $topK, array $apiResult): int
    {
        $matched = $apiResult['matched_input'] ?? [];
        $matchedSourceCode = $matched['id'] ?? null;
        $matchedType = $matched['type'] ?? null;
        $matchedInternalId = null;

        if ($matchedSourceCode && $matchedType) {
            $table = ($matchedType === 'drug') ? 'drugs' : 'diseases';
            $lookupStmt = db()->prepare("SELECT id FROM $table WHERE source_code = :code LIMIT 1");
            $lookupStmt->execute(['code' => $matchedSourceCode]);
            $row = $lookupStmt->fetch();
            if ($row) {
                $matchedInternalId = (int) $row['id'];
            }
        }

        $stmt = db()->prepare(
            'INSERT INTO prediction_requests (user_id, query_type, input_text, matched_entity_id, matched_entity_type, top_k, status, request_ip)
             VALUES (:user_id, :query_type, :input_text, :matched_entity_id, :matched_entity_type, :top_k, :status, :request_ip)'
        );

        $stmt->execute([
            'user_id' => $userId,
            'query_type' => $queryType,
            'input_text' => $inputText,
            'matched_entity_id' => $matchedInternalId,
            'matched_entity_type' => $matchedType,
            'top_k' => $topK,
            'status' => 'success',
            'request_ip' => $_SERVER['REMOTE_ADDR'] ?? '127.0.0.1',
        ]);

        $requestId = (int) db()->lastInsertId();

        $resultStmt = db()->prepare(
            'INSERT INTO prediction_results (request_id, rank_no, result_entity_type, result_entity_id, result_name, score, metadata_json)
             VALUES (:request_id, :rank_no, :result_entity_type, :result_entity_id, :result_name, :score, :metadata_json)'
        );

        foreach (($apiResult['results'] ?? []) as $index => $item) {
            $itemType = $item['type'] ?? 'disease';
            $itemSourceCode = $item['id'] ?? null;
            $itemInternalId = null;

            if ($itemSourceCode) {
                $itemTable = ($itemType === 'drug') ? 'drugs' : 'diseases';
                $itemLookup = db()->prepare("SELECT id FROM $itemTable WHERE source_code = :code LIMIT 1");
                $itemLookup->execute(['code' => $itemSourceCode]);
                $itemRow = $itemLookup->fetch();
                if ($itemRow) {
                    $itemInternalId = (int) $itemRow['id'];
                }
            }

            $resultStmt->execute([
                'request_id' => $requestId,
                'rank_no' => $index + 1,
                'result_entity_type' => $itemType,
                'result_entity_id' => $itemInternalId,
                'result_name' => $item['name'] ?? 'Unknown',
                'score' => $item['score'] ?? 0,
                'metadata_json' => json_encode($item, JSON_UNESCAPED_UNICODE),
            ]);
        }

        return $requestId;
    }
}
