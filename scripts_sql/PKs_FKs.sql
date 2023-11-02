USE sleep;

# CLAVES PRIMARIAS

# Clave primaria de facts

ALTER TABLE facts
MODIFY COLUMN `Person ID` INT NOT NULL,
ADD PRIMARY KEY (`Person ID`);

# Clave primaria de occupation

ALTER TABLE occupation
MODIFY COLUMN `id Occupation` INT NOT NULL,
ADD PRIMARY KEY (`id Occupation`);

# Clave primaria de gender

ALTER TABLE gender
MODIFY COLUMN `id Gender` INT NOT NULL,
ADD PRIMARY KEY (`id Gender`);

# Clave primaria de bmi

ALTER TABLE bmi
MODIFY COLUMN `id BMI Category` INT NOT NULL,
ADD PRIMARY KEY (`id BMI Category`);

# Clave primaria de disorder

ALTER TABLE disorder
MODIFY COLUMN `id Sleep Disorder` INT NOT NULL,
ADD PRIMARY KEY (`id Sleep Disorder`);

# CLAVES FORANEAS

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