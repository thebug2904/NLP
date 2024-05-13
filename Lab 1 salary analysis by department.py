import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv(r"D:\Dataset\Salary_Data.csv")  
department = input("Enter the department to calculate: ")
filtered_df = df[df['department'] == department]

# Calculate the salary statistics
min_salary = filtered_df['Salary'].min()
max_salary = filtered_df['Salary'].max()
avg_salary = filtered_df['Salary'].mean()
median_salary = filtered_df['Salary'].median()

# Print the salary statistics
print("Minimum salary for", department, ":", min_salary)
print("Maximum salary for", department, ":", max_salary)
print("Average salary for", department, ":", avg_salary)
print("Median salary for", department, ":", median_salary)

# Create a box plot to visualize the salary distribution
filtered_df.boxplot(column='Salary', vert=False, showfliers=False)
plt.title("Salary Distribution for " + department)
plt.show()

# Create a bar graph to visualize salary distribution across departments
df.groupby('department')['Salary'].mean().plot(kind='bar')
plt.title("Average Salary by Department")
plt.show()

# Create a pie chart to visualize the proportion of employees in each department
df.groupby('department')['Salary'].sum().plot(kind='pie', autopct="%1.1f%%")
plt.title("Salary Distribution by Department")
plt.show()

#scatter
plt.figure(figsize=(10, 6))
plt.scatter(df['Salary'], df['employee_name'], color='green')
plt.xlabel('Salary')
plt.ylabel('employee_name')
plt.title('Scatter Plot of Salary by Employee Id')
plt.show()

#line graph
df.groupby('department')['Salary'].mean().sort_values().plot(kind='line', marker='o')
plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.grid(True)
plt.show()