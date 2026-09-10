wget https://rasp2.zhanglab.net/api/downloadfile/?filetype=bed&filelist=hg38/icSHAPE_CR_2021_Sun/HeLa_imputed.bed.gz&filelist=hg38/icSHAPE_CR_2021_Sun/HeLa_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_invitro-PDS_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_invitro-PDS_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_invitro_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_invitro_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_PDS_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_PDS_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_score_imputed.bed.gz&filelist=hg38/Keth-seq-2020/HeLa_kethoxal_vs_no-treat_score_imputed.bed.gz
tar -xzf RASP_files.tar.gz

wget http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz
gunzip Homo_sapiens.GRCh37.cdna.all.fa.gz

wget http://ftp.ensembl.org/pub/release-106/gff3/homo_sapiens/Homo_sapiens.GRCh38.106.gff3.gz
gunzip Homo_sapiens.GRCh38.106.gff3.gz

wget https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip
unzip rna_celline.tsv.zip

wget https://g4atlasdata.s3.eu-west-2.amazonaws.com/G4AtlasDownload/IdentifiedRG4s/27571552_1_KPDS_new.validated.gz
gunzip 27571552_1_KPDS_new.validated.gz

wget https://g4atlasdata.s3.eu-west-2.amazonaws.com/G4AtlasDownload/IdentifiedRG4s/27708011_1_S2A.validated.gz
gunzip 27708011_1_S2A.validated.gz

