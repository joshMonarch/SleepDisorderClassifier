use sleep;

ALTER TABLE facts
ADD CONSTRAINT formato_id_person_facts CHECK (`Person ID` > 0),
ADD CONSTRAINT formato_id_bmi_facts CHECK (`id BMI Category` > 0),
ADD CONSTRAINT formato_id_gender_facts CHECK (`id Gender` > 0),
ADD CONSTRAINT formato_id_disorder_facts CHECK (`id Sleep Disorder` > 0),
ADD CONSTRAINT formato_id_occupation_facts CHECK (`id Occupation` > 0),
ADD CONSTRAINT formato_age_facts CHECK (`Age` > 0),
ADD CONSTRAINT formato_sleep_duration_facts CHECK (`Sleep Duration` >= 5.0 AND `Sleep Duration` <= 8.5),
ADD CONSTRAINT formato_quality_sleep_facts CHECK (`Quality of Sleep` > 0 AND `Quality of Sleep` <= 10),
ADD CONSTRAINT formato_phys_act_facts CHECK (`Physical Activity Level` >= 30 AND `Physical Activity Level` <= 90),
ADD CONSTRAINT formato_stress_lvl_facts CHECK (`Stress Level` >= 1 AND `Stress Level` <= 10),
ADD CONSTRAINT formato_heart_rate_facts CHECK (`Heart Rate` >= 65 AND `Heart Rate` <= 86),
ADD CONSTRAINT formato_daily_steps_facts CHECK (`Daily Steps` >= 3000 AND `Daily Steps` <= 10000),
ADD CONSTRAINT formato_sis_press_facts CHECK (`Sistolic pressure` >= 115.0 AND `Sistolic pressure` <= 142.0),
ADD CONSTRAINT formato_dia_press_facts CHECK (`Diastolic pressure` >= 75.0 AND `Diastolic pressure` <= 95.0);

ALTER TABLE gender
ADD CONSTRAINT formato_id_gender CHECK (`id Gender` > 0),
ADD CONSTRAINT formato_gender  CHECK (`Gender` REGEXP '^(Male|Female)$');

ALTER TABLE bmi
ADD CONSTRAINT formato_id_bmi CHECK (`id BMI Category` > 0),
ADD CONSTRAINT formato_bmi_cat CHECK (`BMI Category` REGEXP '^(Normal Weight|Overweight|Obese)$');

ALTER TABLE disorder
ADD CONSTRAINT formato_id_disorder CHECK (`id Sleep Disorder` >= 0),
ADD CONSTRAINT formato_disorder CHECK (`Sleep Disorder` REGEXP '^(None|Sleep Apnea|Insomnia)$');

ALTER TABLE occupation
ADD CONSTRAINT formato_id_occupation CHECK (`id Occupation` > 0),
ADD CONSTRAINT formato_occupation CHECK (`Occupation` REGEXP '^(Nurse|Doctor|Engineer|Lawyer|Teacher|Accountant|Manager|Sales Representative|Salesperson|Scientist|Software Engineer)$');


