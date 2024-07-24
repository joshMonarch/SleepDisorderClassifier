USE sleep;

# PRIMARY KEYS

# Facts primary key

ALTER TABLE facts
MODIFY COLUMN `Person ID` INT NOT NULL,
ADD PRIMARY KEY (`Person ID`);

# Occupation primary key

ALTER TABLE occupation
MODIFY COLUMN `id Occupation` INT NOT NULL,
ADD PRIMARY KEY (`id Occupation`);

# Gender primary key

ALTER TABLE gender
MODIFY COLUMN `id Gender` INT NOT NULL,
ADD PRIMARY KEY (`id Gender`);

# Bmi primary key

ALTER TABLE bmi
MODIFY COLUMN `id BMI Category` INT NOT NULL,
ADD PRIMARY KEY (`id BMI Category`);

# Disorder primary key

ALTER TABLE disorder
MODIFY COLUMN `id Sleep Disorder` INT NOT NULL,
ADD PRIMARY KEY (`id Sleep Disorder`);

# FOREIGN KEYS

# PK disorder -> facts

ALTER TABLE facts
ADD CONSTRAINT fk_facts_disorder
FOREIGN KEY (`id Sleep Disorder`) REFERENCES disorder(`id Sleep Disorder`);

# PK bmi -> facts

ALTER TABLE facts
ADD CONSTRAINT fk_facts_bmi
FOREIGN KEY (`id BMI Category`) REFERENCES bmi(`id BMI Category`);

# PK occupation -> facts

ALTER TABLE facts
ADD CONSTRAINT fk_facts_occupation
FOREIGN KEY (`id Occupation`) REFERENCES occupation(`id Occupation`);

# PK gender -> facts

ALTER TABLE facts
ADD CONSTRAINT fk_facts_gender
FOREIGN KEY (`id Gender`) REFERENCES gender(`id Gender`);