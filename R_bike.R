rm(list=ls())
setwd("C:/Users/Rajashekar/Videos/Project/Bike Rental")
getwd()
bike_data=read.csv("day.csv")
summary(bike_data)
#Data Display
library(kableExtra)
length(colnames(bike_data))
head(bike_data[,1:5]) %>% kable(caption="bike renting(columns:1-5) ",booktabs=TRUE,longtable=TRUE)
head(bike_data[,6:10]) %>% kable(caption="bike renting(columns:1-5) ",booktabs=TRUE,longtable=TRUE)
head(bike_data[,11:16]) %>% kable(caption="bike renting(columns:1-5) ",booktabs=TRUE,longtable=TRUE)
#Libraries required for Data
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
lapply(x, require, character.only = TRUE)
#commit1 addes on R file
#commit2 added on R file
#commit3 added on R file
#commit4 added on R file
 #Edit commit
#missing values new comment
missing_val=data.frame(apply(bike_data,2,function(x){sum(is.na(x))}))
missing_val
missing_val[2]=colnames(bike_data)
row.names(missing_val)=NULL
names(missing_val)[1]="MissingVal"
names(missing_val)[2]='ColumnName'
missing_val=missing_val[,c(2,1)]

#outlier analysis
str(bike_data)
colnames(bike_data)
summary(bike_data)
#
#[1] "instant"    "dteday"     "season"     "yr"         "mnth"       "holiday"    "weekday"    "workingday" "weathersit"
#[10] "temp"       "atemp"      "hum"        "windspeed"  "casual"     "registered" "cnt" 
#
cat_columns=c("season","yr","mnth","holiday","weekday","workingday","weathersit")
for(i in  1:length(cat_columns)){
 boxplot(bike_data[,15]~bike_data[,cat_columns[i]],
        data=bike_data,
        main=paste("box plot of cnt vs",cat_columns[i]),
        xlab=cat_columns[i],
        ylab="Cnt",
        col="gray",border="black")
  }

#Data Distribution
bike_data
hist_data=subset(bike_data,select=c("temp","atemp","hum","windspeed","cnt"))
names(hist_data)[1]="temperature"
names(hist_data)[2]="Feel temperature"
names(hist_data)[3]="Humidity"
names(hist_data)[4]="Wind Speed"
names(hist_data)[5]="Total Bike Rentals"
hist_columns=colnames(hist_data)
colnames(hist_data)
for(i in 1:length(hist_columns[-5])){
  hist(hist_data[,i],col = sample(colours()),main=paste0("histogram for ",hist_columns[i]),xlab = hist_columns[i])
}
#plotting
library(RColorBrewer)
color=c("red","blue","yellow","grey")
for(i in 1:length(hist_columns[-5])){
plot(hist_data[,i],hist_data$`Total Bike Rentals`,
     main=paste0("Histogram for rented bike count vs ",hist_columns[i]),
     xlab = hist_columns[i],
     ylab="rented bike count",
     type="h",
     col=color[i])
}

#correlation analysis for indepedent numeric variables
library(corrgram)
str(bike_data)
numeric_data =subset(bike_data,select =c("temp","atemp","hum","windspeed","cnt"))
correlation=round(cor(numeric_data),2)
correlation
corrgram(numeric_data, order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
ggplot(aes(x=numeric_data$hum,y=numeric_data$windspeed),data = subset(numeric_data))+
         geom_point()+
         geom_smooth(method=lm)+
         theme(legend.position="bottom")+
         labs(y="windspeed",x="humidity")+
         ggtitle("scatter plot of bike rented count vs Humidity")
str(bike_data)
#ANOVA for categorical variables
season_anova=aov(cnt~season,data=bike_data)
summary(season_anova)#season
yr_anova=aov(cnt~yr,data=bike_data)
summary(yr_anova)#yr
month_anova=aov(cnt~mnth,data=bike_data)
summary(month_anova)#mnth
holiday_anova=aov(cnt~holiday,data=bike_data)
summary(holiday_anova)#-holiday
weekday_anova=aov(cnt~weekday,data=bike_data)
summary(weekday_anova)#-weekday
workingDay_anova=aov(cnt~workingday,data=bike_data)
summary(workingDay_anova)#-working day
weatherSit_anova=aov(cnt~weathersit,data=bike_data)
summary(weatherSit_anova)
#MODELLING
#[1] "instant"    "dteday"     "season"     "yr"         "mnth"       "holiday"    "weekday"    "workingday" "weathersit"
#[10] "temp"       "atemp"      "hum"        "windspeed"  "casual"     "registered" "cnt" 
#divide data into train and test data
bike_data_final=subset(bike_data,select=c("season","yr","mnth","weathersit","temp","hum","windspeed","cnt"))
train_index=sample(1:nrow(bike_data_final),0.8*nrow(bike_data_final))
train_data=bike_data_final[train_index,]
test_data=bike_data_final[-train_index,]
#linear regression
linear_model=lm(cnt~.,data=train_data)
summary(linear_model)
predictions=predict(linear_model,test_data[,-8])
Mape = function(act, pre){
  mean(abs((act - pre)/act))*100
}
Mape(test_data[,8],predictions)
vif(linear_model)

# decission tree regression
library(rpart)
dt_model=rpart(cnt~.,data=train_data)
summary(dt_model)
predictions=predict(dt_model,test_data[,-8])
Mape = function(act, pre){
  mean(abs((act - pre)/act))*100
}
Mape(test_data[,8],predictions)

#RandomForest
RF_Model=randomForest(cnt~.,train_data,importance=TRUE,ntree=500)
#extract rules
treeList=RF2List(RF_Model)
exec=extractRules(treeList = treeList,train_data[,-8])
readableRules=presentRules(exec,colnames(train_data))
readableRules[1:2,]

RF_Predictions=predict(RF_Model,test_data[,-8])
test_data$RF_Predictions=predict(RF_Model,test_data[,-8])
test_data
Mape = function(act, pre){
  mean(abs((act - pre)/act))*100
}
Mape(test_data[,8],RF_Predictions)
#KNN Predictions
library(class)
library(BOSTON)
library(FNN)
z=c()

print(z)
for(i in 1:range(100))
{
   z[i]=999
  if(i%%2==1){
    bike_data_final=subset(bike_data,select=c("season","yr","mnth","weathersit","temp","hum","windspeed","cnt"))
    train_index=sample(1:nrow(bike_data_final),0.8*nrow(bike_data_final))
    train_data=bike_data_final[train_index,]
    test_data=bike_data_final[-train_index,]
    print(i)
  KNN_Model=knnreg(train_data[,1:7],train_data$cnt,k=i)
  KNN_Predictions=predict(KNN_Model,test_data[,1:7])
  
    z[i]=Mape(test_data[,8],KNN_Predictions)
   print(z[i])
  
  }
}
print(min(z))
match(min(z),z)
KNN_Model=knnreg(train_data[,1:7],train_data$cnt,k=5)
KNN_Predictions=predict(KNN_Model,test_data[,1:7])
z[i]=Mape(test_data[,8],KNN_Predictions)
df=data.frame(cbind(test_data[,8],KNN_Predictions))
df
z[11]




















