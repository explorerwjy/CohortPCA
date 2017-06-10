#uff<-"AJ_20151202.hardfiltered.716cases.704controls.merged.commonBed.coding.gz.plus.HapMap"

#folder<-"~/server/BreastCancer/AJ/VCF/PCA_HQ/"
#f_ped<-"~/server/BreastCancer/AJ/Result/AJ_case_control.ped"
#suff<-"AJ_20151202.hardfiltered.716cases.704controls.merged.commonBed.coding.HQ.gz.plus.HapMap"

myplot <- function(folder,suff) {
   cont<-read.table("/home/local/ARCS/nz2274/Resources/1000Gneome_2013/source/integrated_call_samples_v3.20130502.ALL.panel",header=1,fill = T,strip.white = T,sep="\t",check.names=F,stringsAsFactors = F)
  
    f_eval<-paste(folder,suff,".eval",sep="")
    f_evec<-paste(folder,suff,".pca.evec",sep="")
   #  ped<-read.table(f_ped)
    
    
    eval <- read.table(f_eval)
    evec1.pc <- round(eval[1,1]/sum(eval)*100,digits=2)
    evec2.pc <- round(eval[2,1]/sum(eval)*100,digits=2)
    evec3.pc <- round(eval[3,1]/sum(eval)*100,digits=2)
    
    evec <- read.table(f_evec)
    
   ###### hapmap ########
     inds<-as.vector(mapply(as.character(evec$V1),FUN = function(x) unlist(strsplit(x,":"))[1]))
    pops<-c()
    for(id in inds){
      index<-which(cont$sample==id)
      if(length(index)<1){
        pops<-c(pops,"InHouse")
      }else{
        pops<-c(pops,cont$pop[which(cont$sample==id)])
      }
    }
    ##### hapmap #####
    
    
    cols<-c("gray","red","dark blue","orange","cyan","violet",colors()[c(367,371,373,375,376,382,417,435,430,425,466,498,551,555,574)])
    pchs<-c(3,4,3,1,2,c(1:6),c(1:6))
    
    ## pops
    #hapmap<-grep("HapMap",evec$V1,ignore.case = T) #which(grep("HapMap",evec$V1,ignore.case = T))
   # local<- grep("HapMap",evec$V1,ignore.case = T,invert = T) # which(grep("HapMap",evec$V1,ignore.case = T,invert = T))
   # a<-unlist(lapply(as.character(evec$V1[hapmap]),function(x)(unlist(strsplit(as.character(x),split = ":")))[1]))
  #  pops<-unlist(lapply(a,function(x) unlist(strsplit(x,".",fixed = T))[2]))
    
    
    pdf(paste(folder,"PCA.pdf",sep=""))
    par(mar = c(5.1, 5.1, 2.1, 2.1))
    plot(evec[,2], evec[,3], 
         xlab=paste("eigenvector1\n",evec1.pc, "% of observed genetic variation", sep=""), 
         ylab=paste("eigenvector2\n",evec2.pc, "% of observed genetic variation", sep=""), 
         main="PCA",
         col=cols[1],
         pch=pchs[1],cex=1 #,type='n'
         # ,xlim=xr,ylim=yr
         # ,  xlim=c(0.005,0.015),
         # ylim=c(0.002,0.012)
    ) 
    pp<-c()
    lnm<-c("Inhouse-Case")
    i=2
    d=0;
    for(c in unique(cont$super_pop)){
      j=1;
      for(p in unique(cont$pop[which(cont$super_pop==c)])){
        index<-which(pops==p)
        pp<-c(pp,index)
        #  d=d+length(index)
        points(evec[index,2], evec[index,3], 
               col=cols[i],
               pch=pchs[j],
               cex=1.2
        )
        #print(c(d,length(index)))
        # print(c)
        #print(p)
        j=j+1;
      }
      i=i+1
      lnm<-c(lnm,c)
    }
    points(evec[which(pops=="InHouse"),2], evec[which(pops=="InHouse"),3], 
           col=cols[1],
           pch=20,
           cex=1.2
    )
    legend("topleft",legend=lnm,fill=cols[1:(length(lnm))],bty='n')
    
    
    ########
    plot(evec[,3], evec[,4], 
         xlab=paste("eigenvector2\n",evec2.pc, "% of observed genetic variation", sep=""), 
         ylab=paste("eigenvector3\n",evec3.pc, "% of observed genetic variation", sep=""), 
         main="PCA",
         col=cols[1],
         pch=pchs[1],cex=1 #,type='n'
         # ,xlim=xr,ylim=yr
         # ,  xlim=c(0.005,0.015),
         # ylim=c(0.002,0.012)
    ) 
    pp<-c()
    lnm<-c("Inhouse-Case")
    i=2
    d=0;
    for(c in unique(cont$super_pop)){
      j=1;
      for(p in unique(cont$pop[which(cont$super_pop==c)])){
        index<-which(pops==p)
        pp<-c(pp,index)
        #  d=d+length(index)
        points(evec[index,3], evec[index,4], 
               col=cols[i],
               pch=pchs[j],
               cex=1.2
        )
        #print(c(d,length(index)))
        # print(c)
        #print(p)
        j=j+1;
      }
      i=i+1
      lnm<-c(lnm,c)
    }
    points(evec[which(pops=="InHouse"),3], evec[which(pops=="InHouse"),4], 
           col=cols[1],
           pch=20,
           cex=1.2
    )
    legend("topleft",legend=lnm,fill=cols[1:(length(lnm))],bty='n')
    
 
    
    dev.off()
}

args<-commandArgs(TRUE)
folder=args[1];
suff=args[2];
myplot(folder,suff)
