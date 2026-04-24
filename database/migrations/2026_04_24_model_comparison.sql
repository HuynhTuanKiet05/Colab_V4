USE `drug_disease_ai`;

CREATE TABLE IF NOT EXISTS `model_comparison_requests` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `user_id` BIGINT UNSIGNED NOT NULL,
  `dataset` VARCHAR(50) NOT NULL,
  `drugs_json` JSON NOT NULL,
  `diseases_json` JSON NOT NULL,
  `top_k` INT NOT NULL DEFAULT 5,
  `status` ENUM('pending', 'success', 'failed') NOT NULL DEFAULT 'success',
  `error_message` TEXT NULL,
  `request_ip` VARCHAR(45) DEFAULT NULL,
  `result_json` JSON DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_model_comparison_user_id` (`user_id`),
  KEY `idx_model_comparison_dataset` (`dataset`),
  KEY `idx_model_comparison_created_at` (`created_at`),
  CONSTRAINT `fk_model_comparison_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS `model_comparison_results` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `request_id` BIGINT UNSIGNED NOT NULL,
  `rank_no` INT NOT NULL,
  `drug_code` VARCHAR(80) NOT NULL,
  `drug_name` VARCHAR(255) NOT NULL,
  `disease_code` VARCHAR(80) NOT NULL,
  `disease_name` VARCHAR(255) NOT NULL,
  `original_score` DECIMAL(10,6) NOT NULL,
  `improved_score` DECIMAL(10,6) NOT NULL,
  `delta_score` DECIMAL(10,6) NOT NULL,
  `winner` ENUM('original', 'improved', 'tie') NOT NULL DEFAULT 'tie',
  `metadata_json` JSON DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_model_comparison_results_request_id` (`request_id`),
  KEY `idx_model_comparison_results_winner` (`winner`),
  CONSTRAINT `fk_model_comparison_results_request` FOREIGN KEY (`request_id`) REFERENCES `model_comparison_requests` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;
