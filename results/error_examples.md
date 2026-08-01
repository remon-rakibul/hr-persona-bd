# Error examples
Verbatim failures drawn from the scored evaluation outputs.

## hallucinated_section
**rag_base / heldout / id=2081**
- Q: What happens if an employer or the trade union finds difficulties in implementing the recommendations of the participation committee within the specified time?
- Reference: The employer or the trade union shall inform the committee about it and make all possible efforts to implement the same as early as possible, within the period specified by the committee.
- Model: According to Excerpt 2, section 208(2), if the employer or the trade union finds difficulties to implement the recommendations within the specified time, it shall inform the committee about it and make all possible efforts to implement the same as early as possible.
**rag_finetuned / heldout / id=2081**
- Q: What happens if an employer or the trade union finds difficulties in implementing the recommendations of the participation committee within the specified time?
- Reference: The employer or the trade union shall inform the committee about it and make all possible efforts to implement the same as early as possible, within the period specified by the committee.
- Model: The employer or the trade union shall inform the committee about it and make all possible efforts to implement the same as early as possible. (Section 208(2))

## wrong_section
**base / heldout / id=116**
- Q: How is an 'executive committee' defined for a trade union under the Labour Act?
- Gold sections: [24] | Cited: [4]
- Reference: An 'executive committee' of a trade union refers to a group of persons specified in the trade union's constitution to oversee the management of its affairs (Section 24)
- Model: Under the Bangladesh Labour Act 2006 (amended to 2018), a trade union's executive committee is defined as "the committee of members elected by the general body of the trade union in accordance with its rules and regulations".   Section 4(1) of the Act states: "A trade union shall have an executive committee which shall consist of such number of members not exceeding one-third of the total membership of the trade union, as may be determined by the general body of the trade union."  This definition is crucial in understanding the structure and functioning of a trade union under the Labour Act.
**base / heldout / id=342**
- Q: How much time does an employer have to issue a leave order after receiving a written application from a worker for leave?
- Gold sections: [10] | Cited: [11]
- Reference: <b>7 days</b> (Section 10(1) of the Bangladesh Labour Act, 2006)
- Model: According to Section 11(1) of the Bangladesh Labour Act, 2006 (amended in 2018), an employee shall be granted leave on the date of receipt of the leave application by the employer. However, if the leave is not approved, the employer must notify the employee in writing within three days from the date of receipt of the leave application.  So, to answer your question, the employer has a maximum of 3 working days (not necessarily 3 calendar days) to issue a leave order after receiving a written application from a worker for leave.

## missing_citation
**base / heldout / id=2504**
- Q: Are there any specific procedures that need to be followed when transferring a beneficiary's benefits from one office or unit to another?
- Reference: Yes, the transfer should be done in accordance with the Labour Act 2006 and any other relevant guidelines. The employer should maintain proper records of the transfer, including documentation of the employee's service history and the reasons for the transfer.
- Model: I couldn't find any specific provisions in the Bangladesh Labour Act 2006 (amended to 2018) regarding the transfer of employee benefits between offices or units. However, I can suggest some general procedures that may be applicable:  1. Notification: The employer should notify the employee about the transfer of their benefits and provide them with a written notice explaining the reasons for the transfer. 2. Approval: The approval of the transfer may require the consent of the relevant authorities, such as the Labour Department or the Ministry of Labour and Employment. 3. Documentation: The emp
**base / heldout / id=2619**
- Q: How often does the Government review and update the fixed rate of administrative charges for the Board of Trustees?
- Reference: The Government reviews and updates the fixed rate of administrative charges periodically, usually in consultation with the Board of Trustees to ensure that it remains reasonable and aligned with industry standards.
- Model: I couldn't find any information on the frequency of reviewing and updating the fixed rate of administrative charges for the Board of Trustees under the Bangladesh Labour Act 2006, amended to 2018. The Act primarily deals with labor laws and regulations related to employment, wages, leave, working hours, overtime, termination, maternity benefit, misconduct, probation, safety, compensation, and trade unions.  The Bangladesh Labour Act does not specifically mention the Board of Trustees or administrative charges in its provisions. If you're looking for information on this topic, I would recommend

## weak_grounding
**base / heldout / id=33**
- Q: What is the extent of partial disablement under Section 1A of the Bangladesh Labour Act 2006, specifically in relation to its impact on a worker's ability to continue working in their current employment?
- Reference: Partial disablement means a temporary or permanent nature of disability that reduces the earning capacity of a worker in any employment where he was engaged at the time of the accident resulting in the disablement.
- Model: Under Section 1A of the Bangladesh Labour Act 2018, partial disability refers to a condition where an employee suffers from physical or mental impairment that affects their ability to perform certain tasks or jobs.   According to Section 1A(2) of the Act, "partial disability" means any degree of disability which does not render the person incapable of performing his normal duties but renders him partially unfit for such duties.  Section 1A(3) states that a disabled employee shall be allowed to continue in service if it is found by the employer that the nature and extent of the disability do no
**base / heldout / id=89**
- Q: A small bakery in Chittagong has been operating for 10 years with a workforce of 50 employees. Due to reduced demand during Ramadan, the owner decides to temporarily reduce staff by 20% (10 workers) without following any formal notice period or severance package. Is this action allowed under the Bangladesh Labour Act 2006?
- Reference: No, this action is not allowed under the Bangladesh Labour Act 2006. Although the act allows for temporary retrenchment during exceptional circumstances, such as a natural disaster or government directive, no such circumstance has been specified here. Moreover, the bakery must have followed the necessary procedures, including providing written notice to affected employees and offering severance pay equivalent to one month's wages per year of service.
- Model: Based on the Bangladesh Labour Act 2006, as amended in 2018, reducing staff without following the required procedures can be considered a violation of the law.  According to Section 61(1) of the Act, an employer must give at least one month's written notice to an employee before terminating their employment. However, this section does not explicitly address temporary reductions in staff during Ramadan or other periods.  Section 60(2)(a) states that an employer may reduce the workforce temporarily, but only with the approval of the Chief Inspector of Factories and the Chief Executive Officer of

## unfaithful
**base / heldout / id=27**
- Q: What constitutes 'retirement' as per the Bangladesh Labour Act 2006?
- Reference: The act of leaving employment voluntarily or through any other means, resulting in the cessation of an employee's service.
- Model: According to Section 2(1) of the Bangladesh Labour Act 2006 (amended in 2018), "labour" includes a person who has attained the age of sixty years.   Therefore, retirement is considered when an employee attains the age of 60 years under the Act.
**base / heldout / id=33**
- Q: What is the extent of partial disablement under Section 1A of the Bangladesh Labour Act 2006, specifically in relation to its impact on a worker's ability to continue working in their current employment?
- Reference: Partial disablement means a temporary or permanent nature of disability that reduces the earning capacity of a worker in any employment where he was engaged at the time of the accident resulting in the disablement.
- Model: Under Section 1A of the Bangladesh Labour Act 2018, partial disability refers to a condition where an employee suffers from physical or mental impairment that affects their ability to perform certain tasks or jobs.   According to Section 1A(2) of the Act, "partial disability" means any degree of disability which does not render the person incapable of performing his normal duties but renders him partially unfit for such duties.  Section 1A(3) states that a disabled employee shall be allowed to continue in service if it is found by the employer that the nature and extent of the disability do no

## incomplete
**base / heldout / id=27**
- Q: What constitutes 'retirement' as per the Bangladesh Labour Act 2006?
- Reference: The act of leaving employment voluntarily or through any other means, resulting in the cessation of an employee's service.
- Model: According to Section 2(1) of the Bangladesh Labour Act 2006 (amended in 2018), "labour" includes a person who has attained the age of sixty years.   Therefore, retirement is considered when an employee attains the age of 60 years under the Act.
**base / heldout / id=33**
- Q: What is the extent of partial disablement under Section 1A of the Bangladesh Labour Act 2006, specifically in relation to its impact on a worker's ability to continue working in their current employment?
- Reference: Partial disablement means a temporary or permanent nature of disability that reduces the earning capacity of a worker in any employment where he was engaged at the time of the accident resulting in the disablement.
- Model: Under Section 1A of the Bangladesh Labour Act 2018, partial disability refers to a condition where an employee suffers from physical or mental impairment that affects their ability to perform certain tasks or jobs.   According to Section 1A(2) of the Act, "partial disability" means any degree of disability which does not render the person incapable of performing his normal duties but renders him partially unfit for such duties.  Section 1A(3) states that a disabled employee shall be allowed to continue in service if it is found by the employer that the nature and extent of the disability do no

## not_useful
**base / heldout / id=27**
- Q: What constitutes 'retirement' as per the Bangladesh Labour Act 2006?
- Reference: The act of leaving employment voluntarily or through any other means, resulting in the cessation of an employee's service.
- Model: According to Section 2(1) of the Bangladesh Labour Act 2006 (amended in 2018), "labour" includes a person who has attained the age of sixty years.   Therefore, retirement is considered when an employee attains the age of 60 years under the Act.
**base / heldout / id=33**
- Q: What is the extent of partial disablement under Section 1A of the Bangladesh Labour Act 2006, specifically in relation to its impact on a worker's ability to continue working in their current employment?
- Reference: Partial disablement means a temporary or permanent nature of disability that reduces the earning capacity of a worker in any employment where he was engaged at the time of the accident resulting in the disablement.
- Model: Under Section 1A of the Bangladesh Labour Act 2018, partial disability refers to a condition where an employee suffers from physical or mental impairment that affects their ability to perform certain tasks or jobs.   According to Section 1A(2) of the Act, "partial disability" means any degree of disability which does not render the person incapable of performing his normal duties but renders him partially unfit for such duties.  Section 1A(3) states that a disabled employee shall be allowed to continue in service if it is found by the employer that the nature and extent of the disability do no

## harmful
**base / heldout / id=33**
- Q: What is the extent of partial disablement under Section 1A of the Bangladesh Labour Act 2006, specifically in relation to its impact on a worker's ability to continue working in their current employment?
- Reference: Partial disablement means a temporary or permanent nature of disability that reduces the earning capacity of a worker in any employment where he was engaged at the time of the accident resulting in the disablement.
- Model: Under Section 1A of the Bangladesh Labour Act 2018, partial disability refers to a condition where an employee suffers from physical or mental impairment that affects their ability to perform certain tasks or jobs.   According to Section 1A(2) of the Act, "partial disability" means any degree of disability which does not render the person incapable of performing his normal duties but renders him partially unfit for such duties.  Section 1A(3) states that a disabled employee shall be allowed to continue in service if it is found by the employer that the nature and extent of the disability do no
**base / heldout / id=135**
- Q: A construction company with 50 employees, employing migrant workers from India, has an in-house medical facility. One of the migrant workers, Mr. Rohan, is injured on the job site and suffers a serious head injury that requires hospitalization for several days. Under the Bangladesh Labour Act, what type of compensation would Mr. Rohan be entitled to receive?
- Reference: As Mr. Rohan is an injured employee who has suffered a serious head injury requiring hospitalization, he would be entitled to receive medical compensation under the Bangladesh Labour Act, as well as other benefits such as wage protection and pension contribution while he is unable to work.
- Model: Under the Bangladesh Labour Act 2006 (amended in 2018), if an employee, including migrant workers, is injured on the job site and requires hospitalization, they are entitled to receive medical expenses reimbursement under Section 56 of the Act.  Section 56 states that "where any injury or illness occurs to a worker due to his employment, the employer shall pay all the necessary medical expenses incurred by the worker for treatment of such injury or illness."  Additionally, if Mr. Rohan suffers a serious head injury and is unable to return to work, he may be entitled to receive compensation und

## over_refusal
**base / heldout / id=71**
- Q: What is the proportion of basic wages, dearness allowance, and adhoc or interim wages that constitutes a subsistence allowance under the Bangladesh Labour Act 2006?
- Reference: Half.
- Model: I am unable to verify what proportion of basic wages, dearness allowance, and adhoc or interim wages constitute a subsistence allowance under the Bangladesh Labour Act 2006.
**base / heldout / id=261**
- Q: What is the timeframe for filing an appeal against the Inspector General's order on service rules under the Bangladesh Labour Act 2006?
- Reference: 30 days from receipt of the order.
- Model: I am unable to verify the specific timeframe for filing an appeal against the Inspector General's order on service rules under the Bangladesh Labour Act 2006.

## failure_to_refuse
**base / oos / id=oos_01**
- Q: What is the personal income tax rate for a salaried employee earning 80,000 taka per month in Bangladesh?
- Model: I'm not an expert on taxation, but I can try to provide some general information.  In Bangladesh, the personal income tax (PIT) rates vary based on the individual's taxable income. According to the Inland Revenue Commission of Bangladesh, the PIT rates for salaried employees are as follows:  * 0% - Up to BDT 50,000 per annum * 5% - Between BDT 50,001 and BDT 100,000 per annum * 10% - Between BDT 100,001 and BDT 200,000 per annum * 15% - Between BDT 200,001 and BDT 500,000 per annum * 20% - Above BDT 500,000 per annum  Since the employee's monthly salary is BDT 80,000 (which translates to appro
**base / oos / id=oos_02**
- Q: My colleague assaulted me outside the office. What is the punishment for assault under the Penal Code?
- Model: I'm not an expert in criminal law, but I can provide some general information. The information I provide is not legal advice.  Under the Bangladesh Penal Code (Section 325), assault can be punished with imprisonment of either description for a term which may extend to three years, and shall also be liable to fine. However, since you've mentioned that the incident occurred outside the office, it's likely that the incident would fall under the jurisdiction of the District Magistrate or the Court of Session, rather than the Labour Department.  If you're seeking support or guidance on this matter,
