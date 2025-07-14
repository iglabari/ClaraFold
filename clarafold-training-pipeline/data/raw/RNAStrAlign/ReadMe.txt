This is a database of structural alignments for 
RNA families and subfamilies collected from RNA databases online. The
structures for each family are categorized into subfamilies (if there are any)
according to the database's original classification and stored in the
respectively named subdirectories inside the directories representing
their parent families. Each subfamily has an alignment inside its own
subdirectory. Only structures that appear both in the alignments and as
structure files in the original database are kept. The families included are:

>1 Transfer RNA (tRNA) family
tRNA structures and alignments are downloaded from tRNAdb 2009 [1]. Because the
structures in this database do not include variable loops. COVE [2] was used to
annotate the variable loops, but not the other base pairs. All the structures
and the alignment are in the "tRNA" directory.

>2 5S ribosomal RNA (5S rRNA) family
5S rRNA structures and alignments are downloaded from 5S Ribosomal RNA Database
[3].  

>3 Group I intron family
Group I intron structures and alignments are downloaded from Group I Intron
Sequence and Structure Database (GISSD) [4]. Structures with P1 and P2 domain
competition are represented by two structure files which either have P1 or
P10 domain and with "_p1" or "_p10"  added to the end of their file names.

>4 RNase P family
RNase P structures and alignments are downloaded from The RNase P Database
[5]. Only 7 subfamilies are kept in our database, the remaining subfamilies are
either not "clean" as the original database stated or have too few
structures. 

>5 Signal recognition particle RNA (SRP RNA) family SRP RNA structures and 
alignments are downloaded from Signal Recognition Particle Database (SRPDB) [6].

>6 Telomerase RNA family
Telomerase RNA structures and alignment are downloaded from Rfam database 
[7]RF00024 family.

>7 Transfer-messenger RNA family
Transfer-messenger RNA structures and alignment are downloaded from tmRNA 
Database [8].


----------------------------------
NOTABLE CHANGES IN RECENT RELEASES

--version 1.1

(1) 5S rRNA database is updated by downloading from the new 5S Ribosomal RNA 
Database [3]. 

(2) A note for ct files in all families: non-canonical base pairs other than 
A-U, G-C, and G-U are removed. 


--version 1.2

(1) Abbreviation codes for degenerate bases (bases other than A, U(T), C, and G) 
in all families are changed into N.


---------------------------------- 

REFERENCES

[1] Jühling, Frank, et al. "tRNAdb 2009: compilation of tRNA sequences and tRNA
genes." Nucleic Acids Research 37.suppl 1 (2009): D159-D162.
[2] http://selab.janelia.org/software.html
[3] Szymanski, Maciej, et al. "5S ribosomal RNA database." Nucleic Acids
Research 30.1 (2002): 176-178.
[4] Zhou, Yu, et al. "GISSD: group I intron sequence and structure database."
Nucleic Acids Research 36.suppl 1 (2008): D31-D37.
[5] Brown, James W. "The ribonuclease P database." Nucleic Acids Research 27.1
(1999): 314-314.
[6] Rosenblad, Magnus Alm, et al. "SRPDB: Signal recognition particle
database." Nucleic Acids Research 31.1 (2003): 363-364.
[7] Nawrocki, Eric P., et al. "Rfam 12.0: updates to the RNA families
database." Nucleic Acids Research (2014): gku1063.
[8] Zwieb, Christian, et al. "tmRDB (tmRNA database)." Nucleic Acids Research
31.1 (2003): 446-447.
