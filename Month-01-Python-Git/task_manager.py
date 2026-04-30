import json
from datetime import datetime


class Task:
    def __init__(self, title, description=""):
        self.title = title
        self.description = description
        self.completed = False
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def mark_complete(self):
        self.completed = True

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        task = cls(data['title'], data.get('description', ''))
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

    def add_task(self, title, description=""):
        task = Task(title, description)
        self.tasks.append(task)
        self.save_tasks()
        print(f"✅ Task added: {title}")

    def list_tasks(self):
        if not self.tasks:
            print("No tasks yet!")
            return
        for i, task in enumerate(self.tasks, 1):
            status = "✅" if task.completed else "⬜"
            print(f"{i}. {status} {task.title} ({task.created_at})")

    def complete_task(self, index):
        try:
            self.tasks[index - 1].mark_complete()
            self.save_tasks()
            print("🎉 Task completed!")
        except IndexError:
            print("Invalid task number!")

    def delete_task(self, index):
        try:
            deleted = self.tasks.pop(index - 1)
            self.save_tasks()
            print(f"🗑️ Deleted: {deleted.title}")
        except IndexError:
            print("Invalid task number!")


def main():
    manager = TaskManager()

    while True:
        print("\n" + "=" * 40)
        print("JRC TASK MANAGER")
        print("=" * 40)
        print("1. Add Task")
        print("2. List Tasks")
        print("3. Complete Task")
        print("4. Delete Task")
        print("5. Exit")

        choice = input("\nChoose an option (1-5): ").strip()

        if choice == "1":
            title = input("Task title: ").strip()
            desc = input("Description (optional): ").strip()
            manager.add_task(title, desc)
        elif choice == "2":
            manager.list_tasks()
        elif choice == "3":
            manager.list_tasks()
            if manager.tasks:
                idx = int(input("\nEnter task number to complete: "))
                manager.complete_task(idx)
        elif choice == "4":
            manager.list_tasks()
            if manager.tasks:
                idx = int(input("\nEnter task number to delete: "))
                manager.delete_task(idx)
        elif choice == "5":
            print("👋 Goodbye JR!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
