library(gbm)

#Working director
setwd("/mnt/Datos/SynologyDrive/Drive/Documentos/Research/Biology_research/Tuberculosis_Helio_2026")

#read data
dat=read.table("BD.csv",sep=";", header = TRUE)


str(dat)

dat$PIROPLASMA=as.factor(dat$PIROPLASMA)
dat$THEILERIA=as.factor(dat$THEILERIA)
dat$ANAPLASMA=as.factor(dat$ANAPLASMA)
dat$SEXO=as.factor(dat$SEXO)
dat$RAZA=as.factor(dat$RAZA)


TB_frame = data.frame(dat$Lesiones_TB, dat$PIROPLASMA,
                       dat$THEILERIA, dat$ANAPLASMA, 
                       dat$VITAMINA_D, dat$CALCIO,
                       dat$PIROPLASMA_Q, dat$EDAD,
                       dat$SEXO, dat$RAZA)
TB_frame_new = TB_frame[complete.cases(TB_frame),]
dim (TB_frame_new)

TB.boost2.tc02.lr0_005 = gbm.step(data = TB_frame, gbm.x = 2:10, gbm.y = 1,family = "bernoulli", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005)


simply=gbm.simplify (TB.boost2.tc02.lr0_005, n.drops=10)
TB.boost2.tc02.lr0_005.simp = gbm.step(data = TB_frame, gbm.x = simply$pred.list [[6]] , gbm.y = 1,family = "bernoulli", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005.simp)

gbm.plot(TB.boost2.tc02.lr0_005.simp,n.plots = 3,, smooth = TRUE, rug = TRUE,write.title = F)




dat=read.table("C:/Users/ejgar/OneDrive/Escritorio/Neobeitar/UNEX/Helio/BD_PATRON.csv",
               sep=";", header = TRUE )


str(dat)

dat$PIROPLASMA=as.factor(dat$PIROPLASMA)
dat$THEILERIA=as.factor(dat$THEILERIA)
dat$ANAPLASMA=as.factor(dat$ANAPLASMA)
dat$SEXO=as.factor(dat$SEXO)
dat$RAZA=as.factor(dat$RAZA)

TB_frame = data.frame(dat$Patron_lesiones, dat$PIROPLASMA,
                      dat$THEILERIA, dat$ANAPLASMA, 
                      dat$VITAMINA_D, dat$CALCIO,
                      dat$PIROPLASMA_Q, dat$EDAD,
                      dat$SEXO, dat$RAZA)
TB_frame_new = TB_frame[complete.cases(TB_frame),]
dim (TB_frame_new)

TB.boost2.tc02.lr0_005 = gbm.step(data = TB_frame, gbm.x = 2:10, gbm.y = 1,family = "bernoulli", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005)


simply=gbm.simplify (TB.boost2.tc02.lr0_005, n.drops=10)
TB.boost2.tc02.lr0_005.simp = gbm.step(data = TB_frame, gbm.x = simply$pred.list [[4]] , gbm.y = 1,family = "bernoulli", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005.simp)

gbm.plot(TB.boost2.tc02.lr0_005.simp,n.plots = 5,, smooth = TRUE, rug = TRUE,write.title = F)





dat=read.table("C:/Users/ejgar/OneDrive/Escritorio/Neobeitar/UNEX/Helio/BD.csv",
               sep=";", header = TRUE )


str(dat)

dat$PIROPLASMA=as.factor(dat$PIROPLASMA)
dat$THEILERIA=as.factor(dat$THEILERIA)
dat$ANAPLASMA=as.factor(dat$ANAPLASMA)
dat$SEXO=as.factor(dat$SEXO)
dat$RAZA=as.factor(dat$RAZA)


TB_frame = data.frame(dat$Score_lesional, dat$PIROPLASMA,
                      dat$THEILERIA, dat$ANAPLASMA, 
                      dat$VITAMINA_D, dat$CALCIO,
                      dat$PIROPLASMA_Q, dat$EDAD,
                      dat$SEXO, dat$RAZA)
TB_frame_new = TB_frame[complete.cases(TB_frame),]
dim (TB_frame_new)

TB.boost2.tc02.lr0_005 = gbm.step(data = TB_frame, gbm.x = 2:10, gbm.y = 1,family = "gaussian", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005)


simply=gbm.simplify (TB.boost2.tc02.lr0_005, n.drops=10)
TB.boost2.tc02.lr0_005.simp = gbm.step(data = TB_frame, gbm.x = simply$pred.list [[7]] , gbm.y = 1,family = "gaussian", tree.complexity = 3,learning.rate = 0.0001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005.simp)

gbm.plot(TB.boost2.tc02.lr0_005.simp,n.plots = 2,, smooth = TRUE, rug = TRUE,write.title = F)



TB_frame = data.frame(dat$IDTC, dat$PIROPLASMA,
                      dat$THEILERIA, dat$ANAPLASMA, 
                      dat$VITAMINA_D, dat$CALCIO,
                      dat$PIROPLASMA_Q, dat$EDAD,
                      dat$SEXO, dat$RAZA)
TB_frame_new = TB_frame[complete.cases(TB_frame),]
dim (TB_frame_new)

TB.boost2.tc02.lr0_005 = gbm.step(data = TB_frame, gbm.x = 2:10, gbm.y = 1,family = "gaussian", tree.complexity = 3,learning.rate = 0.00001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005)


simply=gbm.simplify (TB.boost2.tc02.lr0_005, n.drops=10)
TB.boost2.tc02.lr0_005.simp = gbm.step(data = TB_frame, gbm.x = simply$pred.list [[1]] , gbm.y = 1,family = "gaussian", tree.complexity = 3,learning.rate = 0.00001, bag.fraction = 0.5)
summary (TB.boost2.tc02.lr0_005.simp)

gbm.plot(TB.boost2.tc02.lr0_005.simp,n.plots = 8,, smooth = TRUE, rug = TRUE,write.title = F)



shapiro.test(dat$VITAMINA_D)
t.test(VITAMINA_D~Lesiones_TB, dat)
kruskal.test(VITAMINA_D~Lesiones_TB, dat)
aggregate(VITAMINA_D~Lesiones_TB, dat, mean)

shapiro.test(dat$CALCIO)
t.test(CALCIO~Lesiones_TB, dat)
kruskal.test(CALCIO~Lesiones_TB, dat)
aggregate(CALCIO~Lesiones_TB, dat, mean)

shapiro.test(dat$PIROPLASMA_Q)
t.test(PIROPLASMA_Q~Lesiones_TB, dat)
kruskal.test(PIROPLASMA_Q~Lesiones_TB, dat)
aggregate(PIROPLASMA_Q~Lesiones_TB, dat, mean)


shapiro.test(dat$PIROPLASMA_Q)
t.test(PIROPLASMA_Q~Patron_lesiones, dat)
kruskal.test(PIROPLASMA_Q~Patron_lesiones, dat)
aggregate(PIROPLASMA_Q~Patron_lesiones, dat, mean)

shapiro.test(dat$CALCIO)
t.test(CALCIO~Patron_lesiones, dat)
kruskal.test(CALCIO~Patron_lesiones, dat)
aggregate(CALCIO~Patron_lesiones, dat, mean)

shapiro.test(dat$EDAD)
t.test(EDAD~Patron_lesiones, dat)
kruskal.test(EDAD~Patron_lesiones, dat)
aggregate(EDAD~Patron_lesiones, dat, mean)

shapiro.test(dat$VITAMINA_D)
t.test(VITAMINA_D~Patron_lesiones, dat)
kruskal.test(VITAMINA_D~Patron_lesiones, dat)
aggregate(VITAMINA_D~Patron_lesiones, dat, mean)


prop.table(table(dat$THEILERIA, dat$Patron_lesiones), 2)
chisq.test(dat$THEILERIA, dat$Patron_lesiones)

cor.test(dat$VITAMINA_D, dat$Score_lesional, method="spearman")
cor.test(dat$PIROPLASMA_Q, dat$Score_lesional, method="spearman")


cor.test(dat$PIROPLASMA_Q, dat$IDTC, method="spearman")
cor.test(dat$PIROPLASMA_Q, dat$CALCIO, method="spearman")
cor.test(dat$PIROPLASMA_Q, dat$VITAMINA_D, method="spearman")
cor.test(dat$PIROPLASMA_Q, dat$EDAD, method="spearman")

shapiro.test(dat$IDTC)
t.test(IDTC~THEILERIA, dat)
kruskal.test(IDTC~THEILERIA, dat)
aggregate(IDTC~THEILERIA, dat, mean)

shapiro.test(dat$IDTC)
kruskal.test(IDTC~RAZA, dat)
pairwise.wilcox.test(dat$IDTC, dat$RAZA, "BH")
aggregate(IDTC~RAZA, dat, mean)

shapiro.test(dat$IDTC)
t.test(IDTC~ANAPLASMA, dat)
kruskal.test(IDTC~ANAPLASMA, dat)
aggregate(IDTC~ANAPLASMA, dat, mean)

shapiro.test(dat$IDTC)
t.test(IDTC~PIROPLASMA, dat)
kruskal.test(IDTC~PIROPLASMA, dat)
aggregate(IDTC~PIROPLASMA, dat, mean)
