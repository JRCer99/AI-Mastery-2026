import json
from datetime import datetime, timedelta
from typing import Optional

class Task:
    def __init__(self, title: str, description: str = "", priority: str = "Medium", due_date: Optional[str] = None):
        self.title = title.strip()
        self.description = description.strip()
        self.completed = False
        self.priority = priority.capitalize()  # Low, Medium, High
        self.due_date = due_date
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def mark_complete(self):
        self.completed = True

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict):
        task = cls(
            data['title'],
            data.get('description', ''),
            data.get('priority', 'Medium'),
            data.get('due_date')
        )
        task.completed = data.get('completed', False)
        task.created_at = data.get('created_at')
        return task


class TaskManager:
    def __init__(self, filename="tasks.json"):
        self.filename = filename
        self.tasks = self.load_tasks()

    def load_tasks(self):
        try:
            with open(self.filename, 'r') as f:
                return [Task.from_dict(task) for task in json.load(f)]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_tasks(self):
        with open(self.filename, 'w') as f:
            json.dump([task.to_dict() for task in self.tasks], f, indent=2)

    def add_task(self, title: str, description: str = "", priority: str = "Medium", due_date: Optional[str] = None):
        if not title:
            print("❌ Task title cannot be empty.")
            return
        task = Task(title, description, priority, due_date)
        self.tasks.append(task)
        self.save_tasks()
        print(f"✅ Task added: {title} (Priority: {priority})")

    def list_tasks(self):
        if not self.tasks:
            print("📭 No tasks yet!")
            return
        
        print("\n📋 Your Tasks:")
        for i, task in enumerate(self.tasks, 1):
            status = "✅" if task.completed else "⬜"
            due = f" | Due: {task.due_date}" if task.due_date else ""
            print(f"{i}. {status} [{task.priority}] {task.title}{due}")


def main():
    manager = TaskManager()
    
    while True:
        print("\n" + "="*50)
        print("                 JRC TASK MANAGER")
        print("="*50)
        print("1. Add Task")
        print("2. List Tasks")
        print("3. Complete Task")
        print("4. Delete Task")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ").strip()
        
        if choice == "1":
            title = input("Task title: ").strip()
            desc = input("Description (optional): ").strip()
            priority = input("Priority (Low/Medium/High) [Medium]: ").strip() or "Medium"
            due = input("Due date (YYYY-MM-DD) [optional]: ").strip()
            manager.add_task(title, desc, priority, due if due else None)
            
        elif choice == "2":
            manager.list_tasks()
            
        elif choice == "3":
            manager.list_tasks()
            if manager.tasks:
                try:
                    idx = int(input("\nEnter task number to complete: "))
                    manager.tasks[idx-1].mark_complete()
                    manager.save_tasks()
                    print("🎉 Task marked as complete!")
                except (ValueError, IndexError):
                    print("❌ Invalid task number!")
                    
        elif choice == "4":
            manager.list_tasks()
            if manager.tasks:
                try:
                    idx = int(input("\nEnter task number to delete: "))
                    deleted = manager.tasks.pop(idx-1)
                    manager.save_tasks()
                    print(f"🗑️ Deleted: {deleted.title}")
                except (ValueError, IndexError):
                    print("❌ Invalid task number!")
                    
        elif choice == "5":
            print("👋 See you next time, JRC!")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()